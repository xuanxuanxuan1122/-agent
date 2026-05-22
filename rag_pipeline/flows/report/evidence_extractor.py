from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple
from urllib.parse import urlparse

from rag_pipeline.contracts.evidence_ledger import attach_evidence_ledger

try:
    from rag_pipeline.agents.article_brief import extract_research_subject
except ModuleNotFoundError:  # pragma: no cover - direct script mode fallback.
    from ...agents.article_brief import extract_research_subject


CONTAMINATION_PATTERNS = [
    r"因此可用于支撑：[^。\n]{0,140}的阶段判断、机会排序或风险提示，但必须保留口径边界",
    r"未来\s*\d+[-–—]\d+\s*个月应跟踪该口径是否被更多权威来源[^。\n]{0,180}",
    r"该条数据不能单独证明完整行业趋势[^。\n]{0,140}",
    r"若持续增强，可提高相关章节判断权重",
    r"说明[^。]{0,30}已经有可引用口径[^。\n]{0,80}",
    r"证明强度为[高中低]，需要结合来源类型和时间口径使用",
    r"这条数据把[^。]{0,50}从抽象判断推进到可追踪证据[^。\n]{0,120}",
    r"提供了后续横向比较、建模或访谈验证的锚点",
    r"后续应补充来源范围、样本口径和企业级验证[^。\n]{0,100}",
    r"避免把单点信息误读为稳定趋势",
    r"来源类型为[^。\n]{0,60}，来源为[^。\n]{0,120}，时间口径为[^。\n]{0,80}，证据置信度\d+(?:\.\d+)?",
    r"分析时必须保留冲突而不是裁决",
    r"把[^。\n]{0,80}拆成[^。\n]{0,120}",
    r"已通过\s*IQS\s*获取到联网证据[^。\n]{0,160}",
    r"当前未启用或未成功调用大模型综合分析[^。\n]{0,160}",
    r"先给出可核验的网页结果摘要[^。\n]{0,160}",
    r"关键依据[:：]\s*\d+\.",
]

DIMENSION_ALIASES = {
    "market_size": "市场规模与增速",
    "market": "市场规模与增速",
    "市场规模": "市场规模与增速",
    "市场规模与增速": "市场规模与增速",
    "competition": "竞争格局",
    "竞争格局": "竞争格局",
    "policy": "政策与监管环境",
    "政策与监管": "政策与监管环境",
    "政策与监管环境": "政策与监管环境",
    "technology": "技术路线与产业链",
    "技术路线": "技术路线与产业链",
    "技术路线与产业链": "技术路线与产业链",
    "capital": "投融资与资本动态",
    "资本动态": "投融资与资本动态",
    "投融资与资本市场": "投融资与资本动态",
    "投融资与资本动态": "投融资与资本动态",
}

REPORT_DIMENSIONS = [
    "市场规模与增速",
    "竞争格局",
    "政策与监管环境",
    "技术路线与产业链",
    "投融资与资本动态",
]

DIMENSION_KEYWORDS = {
    "市场规模与增速": {
        "市场规模": 5,
        "规模": 2,
        "cagr": 5,
        "复合年增长": 5,
        "增速": 4,
        "增长率": 4,
        "同比": 3,
        "渗透率": 4,
        "占比": 2,
        "份额": 2,
        "区域": 2,
        "亿美元": 3,
        "亿元": 3,
        "百亿": 3,
        "千亿": 3,
        "产品结构": 2,
    },
    "竞争格局": {
        "竞争": 5,
        "格局": 5,
        "玩家": 5,
        "企业": 2,
        "市场份额": 4,
        "市占率": 4,
        "集中度": 4,
        "头部": 4,
        "龙头": 4,
        "领跑": 3,
        "追赶": 3,
        "护城河": 3,
        "替代": 2,
        "出海": 2,
        "advanced farm": 4,
        "ffrobotics": 4,
        "极飞": 4,
        "约翰迪尔": 4,
        "岚江": 4,
        "abb": 3,
        "发那科": 3,
    },
    "政策与监管环境": {
        "政策": 5,
        "补贴": 5,
        "监管": 5,
        "合规": 5,
        "国债": 4,
        "标准": 4,
        "奖励": 4,
        "试点": 3,
        "示范": 3,
        "政府": 3,
        "农业农村": 3,
        "通知": 3,
        "意见": 3,
        "农事综合服务中心": 4,
        "远程监控": 4,
        "违规经营": 4,
    },
    "技术路线与产业链": {
        "技术": 5,
        "算法": 5,
        "传感器": 5,
        "芯片": 4,
        "通信模块": 4,
        "控制器": 4,
        "大模型": 4,
        "产业链": 5,
        "上游": 4,
        "下游": 3,
        "北斗": 4,
        "定位": 3,
        "误差": 3,
        "作业面积": 3,
        "新能源": 4,
        "感知": 4,
        "执行": 4,
        "控制": 3,
        "采摘": 3,
        "喷灌": 3,
        "集群协同": 4,
    },
    "投融资与资本动态": {
        "融资": 5,
        "投融资": 5,
        "估值": 5,
        "ipo": 5,
        "并购": 5,
        "投资": 4,
        "资本": 4,
        "pe": 4,
        "vc": 4,
        "债券": 4,
        "退出": 4,
        "上市": 4,
        "天使轮": 5,
        "超级轮": 5,
        "资金": 3,
    },
}

SOURCE_TYPE_RANK = {
    "official": 4,
    "policy": 4,
    "research": 3,
    "academic": 3,
    "news": 2,
    "media": 2,
    "unknown": 1,
    "self_media": 0,
}

GENERIC_BAD_FACTS = {
    "",
    "暂无可核验数据",
    "需补充权威来源和可核验口径",
    "需补充具体政策、补贴目录或示范项目文本",
    "需补充融资、投资、并购或估值事件",
    "需补充技术路线、成熟度、试点效果或工程化指标",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _empty_clean_evidence(topic: str = "", *, error: str = "") -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "extractor": "writer_package_clean_evidence_extractor",
        "source": "empty_or_failed_input",
        "evidence_count": 0,
    }
    if error:
        metadata["errors"] = [error]
    return attach_evidence_ledger(
        {
            "topic": str(topic or "").strip(),
            "sources": [],
            "dimensions": {dimension: [] for dimension in REPORT_DIMENSIONS},
            "metadata": metadata,
        }
    )


def _compact_text(value: Any, *, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max(20, max_chars - 3)].rstrip() + "..."


def _source_key(source: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(source.get("url") or source.get("source_url") or "").strip().lower(),
        str(source.get("title") or source.get("source") or "").strip().lower(),
        str(source.get("date") or "").strip(),
    )


def _domain(url: Any) -> str:
    parsed = urlparse(str(url or "").strip())
    return (parsed.netloc or parsed.path.split("/")[0]).removeprefix("www.")


CREDIBILITY_DOMAIN_MAP = {
    ".gov.cn": "A",
    "gov.cn": "A",
    "miit.gov.cn": "A",
    "stats.gov.cn": "A",
    "mof.gov.cn": "A",
    "ndrc.gov.cn": "A",
    "pbc.gov.cn": "A",
    "csrc.gov.cn": "A",
    "cac.gov.cn": "A",
    "samr.gov.cn": "A",
    "caict.ac.cn": "A",
    "caict.com.cn": "A",
    "cas.cn": "A",
    "tsinghua.edu.cn": "A",
    "cninfo.com.cn": "A",
    "sse.com.cn": "A",
    "szse.cn": "A",
    "hkex.com.hk": "A",
    "iresearch.cn": "B",
    "analysys.cn": "B",
    "ccidnet.com": "B",
    "iimedia.cn": "B",
    "qianzhan.com": "B",
    "askci.com": "B",
    "chyxx.com": "B",
    "chinabgao.com": "B",
    "caixin.com": "C",
    "yicai.com": "C",
    "21jingji.com": "C",
    "cls.cn": "C",
    "stcn.com": "C",
    "xinhuanet.com": "C",
    "people.com.cn": "C",
    "cctv.com": "C",
    "chinanews.com": "C",
    "chinadaily.com.cn": "C",
    "36kr.com": "C",
    "thepaper.cn": "C",
    "sina.com.cn": "C",
    "163.com": "C",
    "qq.com": "C",
    "ifeng.com": "C",
    "huxiu.com": "C",
    "sohu.com": "D",
    "baijiahao": "D",
    "guba.eastmoney": "D",
    "zhihu.com": "D",
    "book118.com": "D",
    "renrendoc.com": "D",
    "docin.com": "D",
    "doc88.com": "D",
    "wk.baidu.com": "D",
    "wenku.baidu.com": "D",
}

TITLE_CREDIBILITY_PATTERNS = [
    (re.compile(r"(?:信通院|中国信息通信研究院|社科院|中国科学院|国务院发展研究中心|证监会|工信部|央行|人民银行)"), "A"),
    (re.compile(r"(?:白皮书|蓝皮书|年度报告|招股(?:意向)?书|定期报告|上市公司公告|交易所公告)"), "A"),
    (re.compile(r"(?:艾瑞|易观|赛迪|IDC|Gartner|Forrester|沙利文|Frost\s*&\s*Sullivan|中商产业研究院|智研咨询)"), "B"),
    (re.compile(r"(?:36氪|虎嗅|钛媒体|界面|新浪|腾讯|网易|凤凰|搜狐)"), "C"),
    (re.compile(r"(?:自媒体|公众号|百家号|头条号|股吧|文库)"), "D"),
]

LOW_CREDIBILITY_URL_FRAGMENTS = (
    "caifuhao.eastmoney",
    "guba.eastmoney",
    "mguba.eastmoney",
    "baijiahao",
    "toutiao",
    "zhihu.com",
    "xueqiu.com",
    "weibo.com",
    "sohu.com",
    "book118.com",
    "renrendoc.com",
    "docin.com",
    "doc88.com",
    "wenku.baidu.com",
    "wk.baidu.com",
)

AGGREGATOR_URL_FRAGMENTS = (
    "view.inews.qq.com",
    "kuaixun",
    "finance.sina.com.cn",
    "news.10jqka.com.cn",
    "eastmoney.com/a/",
)


def _infer_credibility(url: str, title: str, source_type: str = "") -> str:
    url_lower = str(url or "").lower()
    title_text = str(title or "")
    source_type_lower = str(source_type or "").lower()
    if source_type_lower in {"self_media", "ugc", "low"} or any(fragment in url_lower for fragment in LOW_CREDIBILITY_URL_FRAGMENTS):
        return "D"
    if any(fragment in url_lower for fragment in AGGREGATOR_URL_FRAGMENTS):
        return "C"
    domain_level = ""
    for domain, level in CREDIBILITY_DOMAIN_MAP.items():
        if domain in url_lower or domain in title_text.lower():
            domain_level = level
            break
    if domain_level == "A":
        return domain_level
    if domain_level == "D":
        return domain_level
    if domain_level in {"B", "C"}:
        return domain_level
    for pattern, level in TITLE_CREDIBILITY_PATTERNS:
        if pattern.search(title_text):
            return level
    if domain_level:
        return domain_level
    if source_type_lower in {"official", "policy", "financial_report"}:
        return "A"
    if source_type_lower in {"research", "academic"}:
        return "B"
    if source_type_lower in {"self_media", "low"}:
        return "D"
    if re.search(r"(通知|意见|方案|规划|指南|公告|决定|办法|条例|标准|白皮书|年报|招股书)", title_text):
        return "A" if re.search(r"(政府|国务院|部|厅|局|交易所|上市公司|公告|年报|招股书)", title_text) else "B"
    return "C"


SCHEMA_LIKE_RE = re.compile(r"^[^。；;]{1,16}[；;][^。；;]{0,16}[；;][^。；;]{0,40}$")


def _normalize_dimension(value: Any) -> str:
    raw = str(value or "").strip()
    if raw in DIMENSION_ALIASES:
        return DIMENSION_ALIASES[raw]
    for key, dimension in DIMENSION_ALIASES.items():
        if key and key in raw:
            return dimension
    return raw


def clean_evidence_text(text: Any) -> str:
    """Remove Analysis/Writer template contamination from evidence text."""

    cleaned = str(text or "")
    cleaned = re.sub(r"EV-[A-Za-z0-9_-]+", "", cleaned)
    cleaned = re.sub(r"【[^】]{1,40}】", "", cleaned)
    cleaned = re.sub(r"\[id:[^\]]+\]", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\[\d{1,3}\]", "", cleaned)
    for pattern in CONTAMINATION_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"[，,；;]\s*[。；;，,]", "。", cleaned)
    cleaned = re.sub(r"[。]{2,}", "。", cleaned)
    cleaned = cleaned.strip(" \t\r\n，,；;。")
    return cleaned


def _is_meaningful_fact(text: str) -> bool:
    stripped = str(text or "").strip()
    if SCHEMA_LIKE_RE.match(stripped):
        return False
    if re.match(r"^年[，,；;。]", stripped):
        return False
    compact = re.sub(r"\s+", "", stripped)
    if len(compact) < 10:
        return False
    if compact in {re.sub(r"\s+", "", item) for item in GENERIC_BAD_FACTS}:
        return False
    if re.fullmatch(r"(?:市场规模|增速|估值|融资金额|并购|资本动态|关键事实)-?\d+(?:\.\d+)?%?", compact):
        return False
    if re.fullmatch(r"[\u4e00-\u9fffA-Za-z]{1,12}\s*-?\d+(?:\.\d+)?", compact):
        return False
    return True


def _dimension_scores(text: str) -> Dict[str, int]:
    normalized = str(text or "").lower()
    scores: Dict[str, int] = {}
    for dimension, keywords in DIMENSION_KEYWORDS.items():
        score = 0
        for keyword, weight in keywords.items():
            if keyword.lower() in normalized:
                score += weight
        scores[dimension] = score
    return scores


def _infer_dimension(text: str, current_dimension: str) -> str:
    scores = _dimension_scores(text)
    best_dimension = max(scores, key=lambda item: scores[item])
    best_score = scores.get(best_dimension, 0)
    current_score = scores.get(current_dimension, 0)

    if not current_dimension:
        return best_dimension if best_score > 0 else ""
    if current_score == 0 and best_score >= 3:
        return best_dimension
    if best_score >= current_score + 3 and best_score >= 4:
        return best_dimension
    return current_dimension


def _source_type_from_item(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    return str(item.get("source_type") or source.get("source_type") or "").strip()


def _source_quality(source_type: str) -> str:
    rank = SOURCE_TYPE_RANK.get(str(source_type or "").strip().lower(), 1)
    if rank >= 4:
        return "high"
    if rank >= 3:
        return "medium"
    if rank >= 1:
        return "normal"
    return "low"


def _clean_metric_value(item: Dict[str, Any], text: str) -> Tuple[str, str]:
    metric = str(item.get("metric") or "").strip()
    value = str(item.get("value") or "").strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        try:
            numeric = abs(float(value))
        except ValueError:
            numeric = 0
        if numeric < 20:
            value = ""
    if value and value not in text and re.fullmatch(r"[\d.]+", value):
        value = ""
    return metric, value


class SourceRegistry:
    def __init__(self) -> None:
        self.sources: List[Dict[str, Any]] = []
        self.index: Dict[Tuple[str, str, str], int] = {}

    def add(self, source: Dict[str, Any]) -> str:
        source = _as_dict(source)
        title = str(source.get("title") or source.get("source") or source.get("source_file") or "").strip()
        url = str(source.get("url") or source.get("source_url") or "").strip()
        date = str(source.get("date") or source.get("time") or "").strip()
        if not title and not url:
            return ""
        source_id = source.get("id") or source.get("source_id")
        source_type = str(source.get("source_type") or source.get("relevance") or "").strip()
        credibility = str(source.get("credibility") or source.get("credibility_level") or "").strip().upper()
        if credibility not in {"A", "B", "C", "D"}:
            credibility = _infer_credibility(url, title, source_type)
        key = _source_key({"title": title, "url": url, "date": date})
        if key in self.index:
            return str(self.index[key])
        numeric_id = int(source_id) if str(source_id or "").isdigit() else len(self.sources) + 1
        while any(int(item.get("id") or 0) == numeric_id for item in self.sources):
            numeric_id += 1
        self.index[key] = numeric_id
        self.sources.append(
            {
                "id": numeric_id,
                "title": title or "未命名来源",
                "url": url,
                "date": date,
                "domain": _domain(url),
                "source_type": source_type,
                "credibility": credibility,
                "credibility_level": credibility,
            }
        )
        return str(numeric_id)


def _seed_sources(pkg: Dict[str, Any], registry: SourceRegistry) -> None:
    writer_report = _as_dict(pkg.get("writer_report"))
    for container_key in ("reformatter_evidence_package", "evidence_package"):
        container = _as_dict(pkg.get(container_key))
        for key in ("sources", "source_registry"):
            for source in _as_list(container.get(key)):
                if isinstance(source, dict):
                    registry.add(source)
    for key in ("sources", "source_registry"):
        for source in _as_list(writer_report.get(key)):
            if isinstance(source, dict):
                registry.add(source)


def _primary_evidence_package(pkg: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(pkg.get("reformatter_evidence_package")) or _as_dict(pkg.get("evidence_package"))


def _strip_report_appendix(markdown: str) -> str:
    return re.split(r"(?m)^##\s*(?:数据来源列表|数据来源|研究口径与来源|附录|参考来源)(?:\s|$|[:：])", str(markdown or ""), maxsplit=1)[0]


def _clean_report_fact_line(line: str) -> str:
    text = str(line or "").strip()
    if not text:
        return ""
    if re.match(r"^\|?\s*:?-{3,}", text):
        return ""
    if text.startswith("|") and text.endswith("|"):
        cells = [cell.strip() for cell in text.strip("|").split("|")]
        cells = [cell for cell in cells if cell and not re.fullmatch(r":?-{3,}:?", cell)]
        text = "；".join(cells)
    text = re.sub(r"^[-*+]\s+", "", text)
    text = re.sub(r"^\d+(?:\.\d+)*[.)、]\s*", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _report_cited_segments(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized or not re.search(r"\[\d{1,3}\]", normalized):
        return []
    parts = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?；;])\s*", normalized)
        if re.search(r"\[\d{1,3}\]", part or "")
    ]
    segments = [part for part in parts if len(part) >= 24]
    return segments or ([normalized] if len(normalized) >= 24 else [])


def _iter_segment_citation_facts(
    text: str,
    *,
    dimension: str,
    seen: Set[Tuple[str, str]],
) -> Iterable[Dict[str, Any]]:
    for segment in _report_cited_segments(text):
        citations = list(dict.fromkeys(re.findall(r"\[(\d{1,3})\]", segment)))
        if not citations:
            continue
        normalized_text = re.sub(r"\s+", " ", segment).strip()
        text_key = re.sub(r"\s+", "", normalized_text.lower())[:260]
        for source_id in citations:
            key = (text_key, source_id)
            if key in seen:
                continue
            seen.add(key)
            yield {
                "text": normalized_text,
                "source_id": source_id,
                "ref": source_id,
                "source_type": "report_citation",
                "credibility_level": "",
                "dimension": dimension,
            }


def _iter_report_cited_facts(pkg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    markdown = _strip_report_appendix(_as_dict(pkg.get("writer_report")).get("report_markdown") or "")
    if not markdown.strip():
        return
    markdown = re.sub(r"(?ms)```.*?```", " ", markdown)
    seen: Set[Tuple[str, str]] = set()
    current_dimension = ""
    for raw_line in markdown.splitlines():
        line = str(raw_line or "").strip()
        heading = re.match(r"^#{2,6}\s+(.+?)\s*$", line)
        if heading:
            current_dimension = heading.group(1).strip()
            continue
        if not re.search(r"\[\d{1,3}\]", line):
            continue
        text = _clean_report_fact_line(line)
        if not text or not re.search(r"\[\d{1,3}\]", text):
            continue
        yield from _iter_segment_citation_facts(text, dimension=current_dimension, seen=seen)
    chunks = re.split(r"\n{2,}", markdown)
    for chunk in chunks:
        raw_chunk = str(chunk or "")
        if re.search(r"(?m)^\s*\|.*\|\s*$", raw_chunk):
            continue
        heading_matches = list(re.finditer(r"(?m)^#{2,6}\s+(.+?)\s*$", raw_chunk))
        chunk_dimension = heading_matches[-1].group(1).strip() if heading_matches else current_dimension
        text = re.sub(r"(?m)^#{1,6}\s*.*$", "", raw_chunk).strip()
        text = re.sub(r"(?m)^\s*\|?\s*:?-{3,}.*$", " ", text)
        text = _clean_report_fact_line(text)
        if not text or not re.search(r"\[\d{1,3}\]", text):
            continue
        yield from _iter_segment_citation_facts(text, dimension=chunk_dimension, seen=seen)


def _iter_raw_evidence(pkg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    evidence_package = _primary_evidence_package(pkg)
    for key in ["clean_evidence_list", "analysis_ready_evidence"]:
        for item in _as_list(evidence_package.get(key)):
            if isinstance(item, dict):
                yield item
    for items in _as_dict(evidence_package.get("chapter_evidence")).values():
        for item in _as_list(items):
            if isinstance(item, dict):
                yield item
    for payload in _as_dict(evidence_package.get("per_dimension")).values():
        if not isinstance(payload, dict):
            continue
        for key in ["clean_facts", "top_evidence"]:
            for item in _as_list(payload.get(key)):
                if isinstance(item, dict):
                    yield item
    for chapter_package in _as_list(pkg.get("chapter_evidence_packages")):
        if not isinstance(chapter_package, dict):
            continue
        for key in ["core_evidence", "supporting_evidence", "sample_evidence", "table_evidence", "clue_evidence"]:
            for item in _as_list(chapter_package.get(key)):
                if isinstance(item, dict):
                    yield item
    for key in ["evidence_list", "merged_evidence"]:
        for item in _as_list(pkg.get(key)):
            if isinstance(item, dict):
                yield item
    yield from _iter_report_cited_facts(pkg)


def _evidence_text(item: Dict[str, Any]) -> str:
    return clean_evidence_text(
        item.get("fact")
        or item.get("fact_text")
        or item.get("text")
        or item.get("clean_fact")
        or item.get("clean_content")
        or item.get("content")
        or _as_dict(item.get("analysis_input")).get("data_point")
    )


def _evidence_time(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    return str(item.get("time") or item.get("period") or item.get("date") or source.get("date") or "").strip()


def _first_text(item: Dict[str, Any], keys: List[str]) -> str:
    source = _as_dict(item.get("source"))
    for key in keys:
        value = item.get(key)
        if value is None:
            value = source.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _evidence_scope(item: Dict[str, Any]) -> str:
    return _first_text(item, ["scope", "market_scope", "geo_scope", "region", "country", "area"])


def _evidence_period(item: Dict[str, Any]) -> str:
    return _first_text(item, ["period", "time_period", "fiscal_period", "year", "date"]) or _evidence_time(item)


def _evidence_unit(item: Dict[str, Any]) -> str:
    return _first_text(item, ["unit", "value_unit", "metric_unit", "currency", "denomination"])


def _evidence_credibility_level(item: Dict[str, Any], source_type: str) -> str:
    source = _as_dict(item.get("source"))
    explicit = str(
        item.get("credibility_level")
        or item.get("credibility")
        or item.get("source_level")
        or source.get("credibility_level")
        or source.get("credibility")
        or source.get("source_level")
        or ""
    ).strip().upper()
    if explicit in {"A", "B", "C", "D"}:
        return explicit
    return _infer_credibility(str(source.get("url") or item.get("url") or ""), str(source.get("title") or item.get("title") or ""), source_type)


def _evidence_source_id(item: Dict[str, Any], registry: SourceRegistry) -> str:
    explicit = item.get("source_id") or item.get("source_ref") or item.get("citation_ref") or item.get("ref")
    if explicit:
        return str(explicit)
    return registry.add(_as_dict(item.get("source")))


def extract_clean_evidence_from_package(pkg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pkg, dict):
        return _empty_clean_evidence(error="writer_package_not_object")
    registry = SourceRegistry()
    _seed_sources(pkg, registry)
    evidence_package = _primary_evidence_package(pkg)
    blueprint = _as_dict(pkg.get("report_blueprint"))
    article_brief = _as_dict(pkg.get("article_brief")) or _as_dict(blueprint.get("article_brief"))
    report_title = str(pkg.get("report_title") or blueprint.get("report_title") or article_brief.get("display_title") or "").strip()
    report_subtitle = str(pkg.get("report_subtitle") or blueprint.get("report_subtitle") or article_brief.get("display_subtitle") or "").strip()
    planning_query = str(
        pkg.get("planning_query")
        or blueprint.get("planning_query")
        or article_brief.get("planning_query")
        or pkg.get("query")
        or evidence_package.get("query")
        or ""
    ).strip()
    research_object = str(pkg.get("research_object") or blueprint.get("research_object") or "").strip()
    if not research_object:
        research_object = extract_research_subject(report_subtitle or planning_query)
    topic = str(pkg.get("topic") or report_subtitle or planning_query or "").strip()
    clean_pkg: Dict[str, Any] = {
        "topic": topic,
        "planning_query": planning_query,
        "article_brief": article_brief,
        "report_title": report_title,
        "report_subtitle": report_subtitle,
        "research_object": research_object,
        "sources": registry.sources,
        "dimensions": {dimension: [] for dimension in REPORT_DIMENSIONS},
        "metadata": {
            "extractor": "writer_package_clean_evidence_extractor",
            "source": "evidence_package_clean_evidence_list",
        },
    }

    seen = set()
    for item in _iter_raw_evidence(pkg):
        text = _compact_text(_evidence_text(item), max_chars=900)
        if not _is_meaningful_fact(text):
            continue
        raw_dimension = item.get("dimension") or item.get("dim")
        dimension = _infer_dimension(text, _normalize_dimension(raw_dimension))
        if not dimension:
            continue
        source_id = _evidence_source_id(item, registry)
        metric, value = _clean_metric_value(item, text)
        source_type = _source_type_from_item(item)
        if not value and not re.search(r"\d", text) and len(text) < 40:
            if source_type != "report_citation" or len(text) < 18:
                continue
        period = _evidence_period(item)
        unit = _evidence_unit(item)
        scope = _evidence_scope(item)
        credibility_level = _evidence_credibility_level(item, source_type)
        key = (dimension, re.sub(r"\s+", "", text.lower())[:220], source_id)
        if key in seen:
            continue
        seen.add(key)
        clean_pkg["dimensions"].setdefault(dimension, []).append(
            {
                "evidence_id": item.get("evidence_id") or item.get("id") or item.get("ref_id"),
                "text": text,
                "source": source_id,
                "source_id": source_id,
                "time": _evidence_time(item),
                "period": period,
                "scope": scope,
                "metric": metric,
                "value": value,
                "unit": unit,
                "credibility_level": credibility_level,
                "source_type": source_type,
                "source_quality": _source_quality(source_type),
                "search_task_id": item.get("search_task_id") or item.get("task_id"),
                "chapter_id": item.get("chapter_id"),
                "claim_id": item.get("claim_id"),
                "hypothesis_id": item.get("hypothesis_id"),
                "proof_role": item.get("proof_role"),
                "evidence_type": item.get("evidence_type"),
                "source_stage": item.get("source_stage") or item.get("origin_query_source"),
            }
        )

    clean_pkg["sources"] = sorted(registry.sources, key=lambda item: int(item.get("id") or 0))
    clean_pkg["metadata"]["evidence_count"] = sum(len(items) for items in clean_pkg["dimensions"].values())
    return attach_evidence_ledger(clean_pkg)


def extract_clean_evidence(writer_package_path: str) -> Dict[str, Any]:
    path = Path(writer_package_path)
    try:
        with path.open("r", encoding="utf-8") as file:
            pkg = json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _empty_clean_evidence(error=f"{type(exc).__name__}: {exc}")
    return extract_clean_evidence_from_package(pkg)
