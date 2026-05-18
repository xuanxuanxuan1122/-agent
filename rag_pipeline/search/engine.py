import argparse
import gc
import json
import os
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..ingest.embedding_qdrant import (
    DEFAULT_ATTN_IMPL,
    DEFAULT_BGE_DENSE_VECTOR_NAME,
    DEFAULT_BGE_M3_BATCH_SIZE,
    DEFAULT_BGE_M3_DEVICE,
    DEFAULT_BGE_M3_MODEL_PATH,
    DEFAULT_BGE_M3_PASSAGE_MAX_LENGTH,
    DEFAULT_BGE_M3_QUERY_MAX_LENGTH,
    DEFAULT_BGE_M3_USE_FP16,
    DEFAULT_BGE_SPARSE_VECTOR_NAME,
    DEFAULT_COLLECTION,
    DEFAULT_DEVICE,
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_DTYPE,
    DEFAULT_ENABLE_SPARSE_VECTORS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_QDRANT_API_KEY,
    DEFAULT_QDRANT_ENABLE_SCALAR_QUANTIZATION,
    DEFAULT_QDRANT_PATH,
    DEFAULT_QDRANT_PREFER_GRPC,
    DEFAULT_QDRANT_URL,
    DEFAULT_SPARSE_VECTOR_NAME,
    BgeM3IngestHelper,
    QwenLocalEmbeddingModel,
    build_sparse_vector,
    collect_runtime_dependency_issues,
    _require_qdrant_client,
    resolve_local_model_dir,
)
from ..config.search_config import (
    DEFAULT_ANSWER_MODE,
    DEFAULT_ANSWER_EVIDENCE_TOP_K,
    DEFAULT_ANSWER_SHOW_EVIDENCE,
    DEFAULT_CANDIDATE_MULTIPLIER,
    DEFAULT_CODE_INSTRUCTION,
    DEFAULT_CONTEXT_HISTORY_TURNS,
    DEFAULT_CORE_EVIDENCE_TOP_K,
    DEFAULT_DENSE_RRF_WEIGHT,
    DEFAULT_EXTERNAL_API_TRUST_ENV,
    DEFAULT_ENABLE_ANSWER_REVIEW,
    DEFAULT_ENABLE_CONTEXTUALIZER,
    DEFAULT_ENABLE_BGE_DENSE_RETRIEVAL,
    DEFAULT_ENABLE_BGE_SPARSE_RETRIEVAL,
    DEFAULT_ENABLE_API_RERANK,
    DEFAULT_ENABLE_LLM_ANSWER_REVIEW,
    DEFAULT_ENABLE_LLM_REFLECTION,
    DEFAULT_ENABLE_LLM_PLANNER,
    DEFAULT_ENABLE_LLM_SYNTHESIS,
    DEFAULT_ENABLE_LOCAL_RERANK,
    DEFAULT_ENABLE_MEMORY,
    DEFAULT_ENABLE_QUERY_PLANNER,
    DEFAULT_ENABLE_REFLECTION,
    DEFAULT_ENABLE_SPARSE_RETRIEVAL,
    DEFAULT_EVIDENCE_TOP_K,
    DEFAULT_FLAT_CHILD_RRF_WEIGHT,
    DEFAULT_KEEP_EMBEDDER_LOADED,
    DEFAULT_LLM_PLANNER_API_KEY,
    DEFAULT_LLM_PLANNER_MAX_QUERIES,
    DEFAULT_LLM_PLANNER_MODEL,
    DEFAULT_LLM_PLANNER_PROVIDER,
    DEFAULT_LLM_PLANNER_TIMEOUT,
    DEFAULT_LLM_PLANNER_URL,
    DEFAULT_LLM_ANSWER_REVIEW_API_KEY,
    DEFAULT_LLM_ANSWER_REVIEW_MODEL,
    DEFAULT_LLM_ANSWER_REVIEW_PROVIDER,
    DEFAULT_LLM_ANSWER_REVIEW_TIMEOUT,
    DEFAULT_LLM_ANSWER_REVIEW_URL,
    DEFAULT_LLM_CONTEXT_DEDUP_THRESHOLD,
    DEFAULT_LLM_CONTEXT_MAX_TOKENS,
    DEFAULT_LLM_CONTEXT_MAX_TOKENS_PER_EVIDENCE,
    DEFAULT_LLM_REFLECTION_API_KEY,
    DEFAULT_LLM_REFLECTION_MODEL,
    DEFAULT_LLM_REFLECTION_PROVIDER,
    DEFAULT_LLM_REFLECTION_TIMEOUT,
    DEFAULT_LLM_REFLECTION_URL,
    DEFAULT_LLM_SYNTHESIS_API_KEY,
    DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING,
    DEFAULT_LLM_SYNTHESIS_MODEL,
    DEFAULT_LLM_SYNTHESIS_PROVIDER,
    DEFAULT_LLM_SYNTHESIS_TIMEOUT,
    DEFAULT_LLM_SYNTHESIS_URL,
    DEFAULT_LOCAL_RERANK_BATCH_SIZE,
    DEFAULT_LOCAL_RERANK_MAX_DOCS,
    DEFAULT_LOCAL_RERANK_MODEL_PATH,
    DEFAULT_MAX_ANSWER_CLAIMS,
    DEFAULT_MAX_PER_DOCUMENT,
    DEFAULT_MEMORY_MAX_TURNS,
    DEFAULT_MEMORY_STORE_DIR,
    DEFAULT_MIN_EVIDENCE,
    DEFAULT_MIN_EVIDENCE_SCORE,
    DEFAULT_PARALLEL_HIERARCHICAL_RETRIEVAL,
    DEFAULT_PARENT_RRF_WEIGHT,
    DEFAULT_PLANNER_HARD_FILTERS,
    DEFAULT_POLICY_INSTRUCTION,
    DEFAULT_PREVIEW_CHARS,
    DEFAULT_QUERY_EXACT,
    DEFAULT_QUERY_HNSW_EF,
    DEFAULT_QUERY_QUANTIZATION_OVERSAMPLING,
    DEFAULT_QUERY_QUANTIZATION_RESCORE,
    DEFAULT_QUERY_VARIANTS,
    DEFAULT_REFLECTION_MAX_HOPS,
    DEFAULT_REFLECTION_OVERLAP_THRESHOLD,
    DEFAULT_RERANK_MAX_CHARS_PER_DOC,
    DEFAULT_RERANK_MAX_DOCS,
    DEFAULT_RERANK_API_KEY,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_PROVIDER,
    DEFAULT_RERANK_TIMEOUT,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_RERANK_URL,
    DEFAULT_RETRIEVAL_INSTRUCTION,
    DEFAULT_RRF_K,
    DEFAULT_SPARSE_RRF_WEIGHT,
    DEFAULT_SUPPORT_EVIDENCE_TOP_K,
    DEFAULT_TOP_K,
    DEFAULT_TRACE_DIR,
    DEFAULT_TRACE_ENABLED,
    DEFAULT_TRACE_TOP_K,
    HF_CACHE_DIR,
)
from .memory import (
    ConversationMemory,
    build_turn_summary,
    contextualize_query,
    generate_turn_id,
    normalize_openai_compatible_chat_url,
)
from .reflection import evidence_chunk_uids, evidence_overlap_ratio, reflect_on_evidence
from .review import review_answer_with_fallback
from .models import (
    AnswerSynthesis,
    EvidenceItem,
    QueryPlan,
    SearchCandidate,
    Turn,
)
from .synthesis import synthesize_answer_with_fallback
from .trace import TraceRecorder

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_ROOT = Path(__file__).resolve().parents[2]

_QWEN_EMBEDDER_CACHE: Dict[tuple, QwenLocalEmbeddingModel] = {}
_BGE_M3_HELPER_CACHE: Dict[tuple, BgeM3IngestHelper] = {}
_LOCAL_RERANKER_CACHE: Dict[str, Any] = {}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ERROR_CODE_RE = re.compile(r"\b[a-zA-Z]{2,}[\-_]?[0-9]{2,}\b|\b[0-9]{3,6}\b")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def unique_preserve(items: Sequence[str], max_items: Optional[int] = None) -> List[str]:
    values: List[str] = []
    seen = set()
    for item in items:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        key = normalize_text(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        values.append(cleaned)
        if max_items and len(values) >= max_items:
            break
    return values


def cleanup_modal_suffix(term: str) -> str:
    cleaned = str(term or "").strip()
    if cleaned.endswith(("能能", "会会")):
        return cleaned[:-1]
    return cleaned


_QUERY_BREAK_RE = re.compile(
    r"(?:可以|能够|如何|怎么|怎样|哪些|是否|需要|确保|以及|和|与|的|在|对|通过|从|因此|如果|因为|所以|而且|并且|并|主要|体现在|方面)"
)
_PUNCT_SPLIT_RE = re.compile(r"[，。！？、；;:：（）【】\[\]{}<>《》“”‘’\s,/\\|]+")
_API_HINT_RE = re.compile(r"(?:api|sdk|http|grpc|sql|curl|docker|k8s|kubernetes|接口|报错|错误码|参数|配置|函数|类|方法)", re.I)
_POLICY_HINT_RE = re.compile(r"(?:制度|规范|流程|审批|合规|权限|准入|要求|标准|章程|sop|policy)", re.I)
_TIME_TOKEN_RE = re.compile(
    r"(?:\b20\d{2}(?:[-/.]\d{1,2}(?:[-/.]\d{1,2})?)?\b|\b\d{4}年\d{1,2}月(?:\d{1,2}日)?\b|\b\d{1,2}月\d{1,2}日\b|截至\S+|最新|当前|今年|去年|本月|上月)"
)
_MULTI_HOP_RE = re.compile(r"(?:以及|同时|并且|并|对比|区别|差异|关联|关系|除了.*还|vs\.?|versus)", re.I)
_NUMERIC_QUERY_RE = re.compile(r"(?:多少|占比|比例|金额|数量|期限|时长|日期|时间|版本|截止|截至|rate|count|version|date)", re.I)
_NUMERIC_TOKEN_RE = re.compile(r"(?:\b\d+(?:\.\d+)?%?\b|\b20\d{2}(?:[-/.]\d{1,2}(?:[-/.]\d{1,2})?)?\b|\b\d{4}年\d{1,2}月(?:\d{1,2}日)?\b|\b\d{1,2}月\d{1,2}日\b)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")
_LEADING_REFERENCE_RE = re.compile(r"^(?:它|其|该公司|该行业|该产品|该业务|该技术|该市场|该赛道)的?")

_GENERIC_QUERY_STOP_TOKENS = {
    "市场",
    "规模",
    "份额",
    "趋势",
    "对比",
    "区别",
    "差异",
    "定义",
    "概念",
    "流程",
    "步骤",
    "原因",
    "现状",
    "情况",
    "主要",
    "方面",
    "体现在",
    "优势",
    "劣势",
    "如何",
    "怎么",
    "为什么",
    "是否",
    "那些",
    "哪些",
    "什么",
    "最新",
    "当前",
    "分析",
    "行研",
    "行业研究",
}

_TASK_DOC_TYPE_HINTS: Dict[str, List[str]] = {
    "market": ["report", "research", "industry", "statistics", "analysis", "financial", "forecast"],
    "comparison": ["report", "research", "analysis", "manual", "faq", "benchmark"],
    "definition": ["guide", "manual", "report", "summary"],
    "trend": ["report", "research", "analysis", "statistics", "forecast"],
    "procedure": ["guide", "faq", "manual", "procedure", "sop"],
    "root_cause": ["analysis", "report", "research", "incident"],
    "status": ["report", "research", "analysis", "update"],
    "fact": ["report", "guide", "manual", "analysis"],
}

_TASK_CHUNK_TYPE_HINTS: Dict[str, List[str]] = {
    "market": ["summary", "table", "list", "heading"],
    "comparison": ["summary", "table", "list", "faq"],
    "definition": ["summary", "heading", "title"],
    "trend": ["summary", "table", "list"],
    "procedure": ["faq", "list", "heading", "summary"],
    "root_cause": ["summary", "faq", "heading"],
    "status": ["summary", "heading", "list"],
    "fact": ["summary", "heading", "faq"],
}

_TASK_NEGATIVE_DOC_TYPES: Dict[str, List[str]] = {
    "market": ["interview", "promotion", "case", "demo", "news"],
    "comparison": ["interview", "promotion", "demo"],
    "definition": ["interview", "promotion"],
    "trend": ["interview", "promotion", "demo"],
    "procedure": ["interview", "promotion", "news"],
    "root_cause": ["promotion", "demo"],
    "status": ["promotion", "demo"],
    "fact": ["promotion", "demo"],
}

_TASK_NEGATIVE_TERMS: Dict[str, List[str]] = {
    "market": ["融资", "路演", "采访", "演示", "案例分享", "愿景", "生态合作", "领导致辞"],
    "comparison": ["融资", "采访", "愿景", "生态合作"],
    "definition": ["采访", "致辞", "愿景"],
    "trend": ["融资", "采访", "案例分享", "领导致辞"],
    "procedure": ["市场规模", "竞争格局", "路演", "采访"],
    "root_cause": ["领导致辞", "愿景", "案例分享"],
    "status": ["采访", "愿景", "生态合作"],
    "fact": ["采访", "愿景", "致辞"],
}

_TASK_FACET_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "market": {
        "scale": ["市场规模", "市场空间", "规模", "销量", "销售额", "出货量", "需求量", "产量", "装机量"],
        "trend": ["趋势", "增速", "增长", "同比", "环比", "cagr", "复合增长", "变化", "渗透率"],
        "landscape": ["份额", "竞争格局", "厂商", "龙头", "区域格局", "集中度", "壁垒"],
        "value_chain": ["产业链", "价值链", "上游", "中游", "下游", "供应链", "渠道"],
        "profitability": ["价格", "单价", "成本", "毛利率", "利润率", "盈利", "商业模式", "收入", "营收"],
        "risk": ["风险", "瓶颈", "挑战", "不确定", "替代", "政策"],
        "source": ["报告", "数据来源", "统计", "口径", "样本", "测算"],
    },
    "comparison": {
        "difference": ["区别", "差异", "不同", "对比", "相比", "优劣"],
        "dimension": ["性能", "成本", "优势", "劣势", "特点", "维度"],
        "scenario": ["适用", "场景", "适合", "推荐", "限制"],
    },
    "definition": {
        "definition": ["是指", "定义", "指的是", "概念", "是什么"],
        "feature": ["特征", "特点", "核心", "本质"],
        "example": ["包括", "例如", "范围", "场景", "示例"],
    },
    "trend": {
        "trend": ["趋势", "增长", "下降", "变化", "走向", "增速", "变革", "革命", "产业革命", "产业变革", "智能技术"],
        "time": ["今年", "去年", "未来", "预计", "季度", "月", "当前"],
        "driver": ["驱动", "驱动力", "因素", "原因", "受益", "影响", "推动", "引领", "主导力量", "核心驱动力", "技术创新", "政策", "需求"],
        "adoption": ["应用", "落地", "商业化", "渗透", "场景", "客户", "产业链"],
        "risk": ["风险", "瓶颈", "挑战", "不确定", "约束", "替代"],
    },
    "procedure": {
        "steps": ["步骤", "流程", "先", "然后", "最后", "操作", "办理"],
        "requirements": ["要求", "条件", "材料", "权限", "前提", "输入", "配置"],
        "caution": ["注意", "风险", "校验", "检查", "失败", "异常", "回滚"],
    },
    "root_cause": {
        "cause": ["原因", "导致", "由于", "因为", "根因", "驱动因素"],
        "mechanism": ["影响", "机制", "链路", "路径", "逻辑", "依据"],
        "evidence": ["数据", "案例", "证据", "表现", "结果"],
    },
    "status": {
        "status": ["现状", "当前", "目前", "现阶段", "最新", "进展", "情况"],
        "problem": ["问题", "瓶颈", "挑战", "风险", "进度", "落地"],
        "time_source": ["截至", "今年", "当前", "最新", "报告", "数据"],
    },
    "fact": {
        "core": ["主要", "包括", "体现在", "信息", "数据", "说明"],
    },
}

_TASK_MIN_COVERAGE: Dict[str, int] = {
    "market": 2,
    "comparison": 2,
    "definition": 1,
    "trend": 2,
    "procedure": 2,
    "root_cause": 2,
    "status": 2,
    "fact": 1,
}

_HARD_FILTER_HINT_KEYS = {"doc_type", "chunk_type"}


def split_sentences(text: str) -> List[str]:
    compact = " ".join(str(text or "").split())
    if not compact:
        return []
    pieces = [piece.strip() for piece in _SENTENCE_SPLIT_RE.split(compact) if piece.strip()]
    return pieces or [compact]


def format_citation_label(citation: Dict[str, str]) -> str:
    doc_title = str(citation.get("doc_title") or "").strip()
    section_title = str(citation.get("section_title") or "").strip()
    chunk_uid = str(citation.get("chunk_uid") or "").strip()
    if doc_title and section_title:
        return f"{doc_title} / {section_title}"
    if doc_title:
        return doc_title
    if section_title:
        return section_title
    return chunk_uid or str(citation.get("source_file") or "").strip()


def build_instruction_query(query: str, intent: str, disable_instruction: bool = False) -> str:
    query = str(query or "").strip()
    if disable_instruction or not query:
        return query
    if intent == "policy":
        return DEFAULT_POLICY_INSTRUCTION.format(query=query)
    if intent == "technical":
        return DEFAULT_CODE_INSTRUCTION.format(query=query)
    return DEFAULT_RETRIEVAL_INSTRUCTION.format(query=query)


def expand_query_variants(query: str, max_variants: int) -> List[str]:
    query = str(query or "").strip()
    if not query:
        return []
    variants: List[str] = [query]
    terms = build_query_terms(query)
    if len(terms) >= 2:
        variants.append(" ".join(terms[: min(6, len(terms))]))
    key_terms = [t for t in terms if len(t) >= 2][:4]
    if key_terms:
        variants.append(" ".join(key_terms))
    if _ERROR_CODE_RE.search(query):
        variants.extend(_ERROR_CODE_RE.findall(query))

    deduped: List[str] = []
    seen = set()
    for item in variants:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
        if len(deduped) >= max(1, max_variants):
            break
    return deduped


def count_cjk(text: str) -> int:
    return len(_CJK_RE.findall(str(text or "")))


def contains_any(text: str, tokens: Sequence[str]) -> bool:
    lowered = normalize_text(text)
    return any(token and token in lowered for token in tokens)


def cleanup_retrieval_text(text: str) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return ""

    compact = re.sub(r"(?i)\bsource_ref:\s*chunk\s*=\s*\d+\s*span\s*=\s*chunk\s*\d+\s*header_path:[^；;。]*[；;]?\s*", "", compact)
    compact = re.sub(r"(?i)\bsection:\s*[^；;。]*[；;]?\s*", "", compact)
    compact = re.sub(r"(?i)\bcontent type:\s*[a-z_]+\s*", "", compact)
    compact = re.sub(r"(?i)\bnumbers:[^；;。]*\blogic_text:\s*", "", compact)
    compact = re.sub(r"(?i)\blogic_text:\s*", "", compact)

    parts = [part.strip() for part in compact.split("|") if part.strip()]
    if len(parts) > 1:
        filtered_parts: List[str] = []
        for part in parts:
            lowered = normalize_text(part)
            if not lowered:
                continue
            if lowered in {"logic", "body", "example", "recommendation", "comparison", "causal", "title", "summary", "table"}:
                continue
            if lowered.startswith("content type:"):
                continue
            if re.fullmatch(r"第\s*\d+\s*页", part):
                continue
            if re.match(r"^[A-Za-z]:\\", part):
                continue
            if count_cjk(part) < 2 and len(part) <= 18:
                continue
            filtered_parts.append(part)
        if filtered_parts:
            compact = " ".join(filtered_parts)

    compact = re.sub(r"(?i)^\s*logic\b\s*", "", compact)
    compact = re.sub(r"(?i)^\s*content type:\s*[^|>:/\\]+", "", compact)
    compact = re.sub(r"(?i)^\s*pptx?\s*逻辑文本\s*>\s*", "", compact)
    compact = re.sub(r"^\s*第\s*\d+\s*页[:：]?\s*", "", compact)
    compact = re.sub(r"^[A-Za-z]:\\[^|>]+", "", compact)
    compact = re.sub(r"(?is)^.*?\bdocument:\s*.*?\.(?:logic|pptx?)\s*", "", compact)
    compact = re.sub(r"(?is)^.*?\bdocument:\s*.*?\bcontent type:\s*[a-z_]+\s*", "", compact)
    compact = re.sub(r"(?is)^.*?\bpptx?\s*逻辑文本\s*>\s*", "", compact)

    prefix_noise = re.search(r"(?i)\b(logic|content type:|pptx?\s*逻辑文本|d:\\)\b", compact[:120])
    if prefix_noise:
        chinese_match = _CJK_RE.search(compact)
        if chinese_match:
            compact = compact[chinese_match.start():]

    compact = re.sub(r"\s+\?\s+", "；", compact)
    compact = re.sub(r"[；;]{2,}", "；", compact)
    compact = re.sub(r"\s{2,}", " ", compact).strip(" |;；")
    if not compact:
        return ""

    english_word_count = len(re.findall(r"[A-Za-z]{3,}", compact))
    if count_cjk(compact) == 0 and english_word_count < 3 and not re.search(r"\d", compact):
        return ""
    return compact


def classify_query_intent(query: str) -> str:
    if _POLICY_HINT_RE.search(query):
        return "policy"
    if _API_HINT_RE.search(query) or _ERROR_CODE_RE.search(query):
        return "technical"
    return "general"


def extract_time_terms(query: str) -> List[str]:
    return unique_preserve(_TIME_TOKEN_RE.findall(str(query or "")))


def infer_evidence_focus(query: str, intent: str, task_type: str) -> List[str]:
    focus: List[str] = []
    lowered = normalize_text(query)
    if task_type == "procedure":
        focus.extend(["procedure", "checklist"])
    elif task_type == "market":
        focus.extend(["statistics", "trend", "competitive_landscape", "value_chain", "profitability", "risk"])
    elif task_type == "comparison":
        focus.extend(["comparison", "structured_list", "rationale"])
    elif task_type == "definition":
        focus.extend(["definition", "summary"])
    elif task_type == "trend":
        focus.extend(["trend", "statistics", "driver", "commercialization", "risk"])
    elif task_type == "root_cause":
        focus.extend(["rationale", "causal"])
    elif task_type == "status":
        focus.extend(["status", "summary"])

    if contains_any(lowered, ["要求", "规范", "制度", "政策", "合规", "policy", "requirement"]):
        focus.append("policy_clause")
    if contains_any(lowered, ["错误", "报错", "异常", "error", "exception", "api", "sdk", "配置", "参数"]):
        focus.append("technical_fix")
    if contains_any(lowered, ["表", "清单", "列表", "table", "list"]):
        focus.append("structured_list")
    if contains_any(lowered, ["市场", "规模", "市场空间", "份额", "销量", "出货量", "渗透率", "需求", "供给"]):
        focus.append("statistics")
    if contains_any(lowered, ["毛利率", "利润率", "盈利", "成本", "价格", "单价", "商业模式"]):
        focus.append("profitability")
    if contains_any(lowered, ["产业链", "价值链", "上游", "中游", "下游", "供应链"]):
        focus.append("value_chain")
    if contains_any(lowered, ["风险", "瓶颈", "挑战", "不确定"]):
        focus.append("risk")
    if intent == "policy":
        focus.append("policy_clause")
    if intent == "technical":
        focus.append("technical_fix")
    if not focus:
        focus.append("fact")
    return unique_preserve(focus)


def infer_filter_hints(query: str, intent: str, task_type: str) -> Dict[str, List[str]]:
    lowered = normalize_text(query)
    hints: Dict[str, List[str]] = {}
    doc_types: List[str] = []
    chunk_types: List[str] = []
    negative_doc_types = list(_TASK_NEGATIVE_DOC_TYPES.get(task_type, []))
    negative_chunk_types: List[str] = []

    if intent == "policy":
        doc_types.extend(["policy", "procedure", "rule", "sop"])
    elif intent == "technical":
        doc_types.extend(["technical", "api", "code", "manual"])

    doc_types.extend(_TASK_DOC_TYPE_HINTS.get(task_type, []))
    chunk_types.extend(_TASK_CHUNK_TYPE_HINTS.get(task_type, []))

    if "faq" in lowered or "常见问题" in lowered:
        chunk_types.append("faq")
    if contains_any(lowered, ["摘要", "总结", "概述", "summary", "overview"]):
        chunk_types.append("summary")
    if contains_any(lowered, ["标题", "章节", "目录", "section", "heading"]):
        chunk_types.extend(["title", "heading"])
    if contains_any(lowered, ["表", "清单", "table", "list"]):
        chunk_types.extend(["table", "list"])
    if task_type == "market":
        negative_chunk_types.extend(["faq"])
    if task_type == "procedure":
        negative_doc_types.extend(["report"])

    if doc_types:
        hints["doc_type"] = unique_preserve(doc_types)
    if chunk_types:
        hints["chunk_type"] = unique_preserve(chunk_types)
    if negative_doc_types:
        hints["negative_doc_type"] = unique_preserve(negative_doc_types)
    if negative_chunk_types:
        hints["negative_chunk_type"] = unique_preserve(negative_chunk_types)
    return hints


def build_body_match_terms(query: str) -> List[str]:
    normalized = normalize_text(query)
    if not normalized:
        return []
    candidates = [normalized]
    candidates.extend(build_query_terms(normalized))
    candidates.extend(extract_subject_terms(normalized))
    candidates.extend(part.strip() for part in re.split(r"[与和及、/]", normalized) if len(part.strip()) >= 2)
    deduped: List[str] = []
    seen = set()
    for item in candidates:
        key = normalize_text(item)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def compute_body_support_score(query: str, payload: Dict[str, Any]) -> float:
    body = normalize_text(payload.get("retrieval_text", "") or payload.get("text", "") or "")
    if not body:
        return 0.0

    normalized_query = normalize_text(query)
    terms = build_body_match_terms(query)
    score = 0.0
    if normalized_query and normalized_query in body:
        score += 0.45

    hits = sum(1 for term in terms if term in body)
    if hits:
        score += min(0.36, 0.14 * hits)
        score += min(0.19, 0.19 * (hits / max(len(terms), 1)))

    return min(1.0, score)


def compute_lexical_bonus(query: str, payload: Dict[str, Any]) -> float:
    normalized_query = normalize_text(query)
    section_title = normalize_text(payload.get("section_title", ""))
    header_text = normalize_text(" ".join(str(v) for v in payload.get("header_path", []) if str(v).strip()))
    text = normalize_text(payload.get("retrieval_text", "") or payload.get("text", ""))
    doc_title = normalize_text(payload.get("doc_title", ""))
    source_file = normalize_text(payload.get("source_file", ""))
    terms = build_query_terms(normalized_query)
    body_support_score = compute_body_support_score(query, payload)

    bonus = 0.0
    if section_title == normalized_query:
        bonus += 0.35
    elif normalized_query and normalized_query in section_title:
        bonus += 0.25
    elif normalized_query and normalized_query in header_text:
        bonus += 0.18
    elif normalized_query and normalized_query in doc_title:
        bonus += 0.16
    elif normalized_query and normalized_query in text:
        bonus += 0.12

    title_hits = sum(1 for t in terms if t in section_title)
    header_hits = sum(1 for t in terms if t in header_text)
    doc_hits = sum(1 for t in terms if t in doc_title or t in source_file)
    body_hits = sum(1 for t in terms if t in text)

    bonus += min(0.30, 0.10 * title_hits)
    bonus += min(0.18, 0.06 * header_hits)
    bonus += min(0.10, 0.04 * doc_hits)
    bonus += min(0.08, 0.015 * body_hits)
    if title_hits > 0 and len(terms) >= 2:
        bonus += 0.05
    if body_support_score < 0.12 and (title_hits > 0 or header_hits > 0):
        bonus = min(bonus, 0.22)
    return bonus

def compute_metadata_bonus(item: Dict[str, Any]) -> float:
    bonus = 0.0
    quality_score = float(item.get("quality_score", 0.0) or 0.0)
    info_density = float(item.get("info_density", 0.0) or 0.0)
    noise_score = float(item.get("noise_score", 0.0) or 0.0)
    chunk_type = normalize_text(item.get("chunk_type", ""))
    section_kind = normalize_text(item.get("section_kind", ""))
    quality_flags = {normalize_text(flag) for flag in item.get("quality_flags", []) if normalize_text(flag)}
    if quality_score > 0:
        bonus += min(0.08, quality_score * 0.08)
    if info_density > 0:
        bonus += min(0.07, info_density * 0.07)
    if noise_score > 0:
        bonus -= min(0.10, noise_score * 0.10)
    if chunk_type in {"faq", "title", "heading", "summary", "abstract"}:
        bonus += 0.05
    if item.get("section_title"):
        bonus += 0.02
    if section_kind in {"publication", "toc", "catalog", "cover", "member_list", "promo_page", "endorsement", "preface"}:
        bonus -= 0.06
    if {"publication_metadata", "toc_section", "catalog_page", "cover_page", "member_list", "promotional_page", "visual_noise_heavy", "low_info_density", "endorsement_like", "preface_like"} & quality_flags:
        bonus -= 0.08
    return bonus


def build_search_params(
    qmodels: Any,
    hnsw_ef: int,
    exact: bool,
    quantization_rescore: bool,
    quantization_oversampling: float,
):
    quantization = None
    if DEFAULT_QDRANT_ENABLE_SCALAR_QUANTIZATION:
        quantization = qmodels.QuantizationSearchParams(
            rescore=quantization_rescore,
            oversampling=max(1.0, float(quantization_oversampling or 1.0)),
        )
    return qmodels.SearchParams(
        hnsw_ef=hnsw_ef if hnsw_ef > 0 else None,
        exact=exact,
        quantization=quantization,
    )


def rrf(rank: int, k: int = DEFAULT_RRF_K, weight: float = 1.0) -> float:
    return float(weight) / (k + rank)


def rrf_weight_for_source(source_name: str, score_key: str = "") -> float:
    source = normalize_text(" ".join([source_name or "", score_key or ""]))
    if "flat_child" in source:
        return DEFAULT_FLAT_CHILD_RRF_WEIGHT
    if "parent" in source:
        return DEFAULT_PARENT_RRF_WEIGHT
    if "sparse" in source or "lexical" in source:
        return DEFAULT_SPARSE_RRF_WEIGHT
    return DEFAULT_DENSE_RRF_WEIGHT


def weighted_rrf(rank: int, source_name: str, score_key: str = "") -> float:
    return rrf(rank, weight=rrf_weight_for_source(source_name, score_key))


def build_client(db_path: str, url: str, api_key: str, prefer_grpc: bool):
    QdrantClient, _ = _require_qdrant_client()
    try:
        timeout = float(os.getenv("QDRANT_CLIENT_TIMEOUT_SECONDS", "30") or 30)
    except (TypeError, ValueError):
        timeout = 30.0
    if url:
        kwargs = {"url": url, "prefer_grpc": prefer_grpc, "check_compatibility": False, "timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        return QdrantClient(**kwargs)
    if db_path == ":memory:":
        try:
            return QdrantClient(":memory:", timeout=timeout)
        except TypeError:
            return QdrantClient(":memory:")
    try:
        return QdrantClient(path=db_path, timeout=timeout)
    except TypeError:
        return QdrantClient(path=db_path)


def probe_local_qdrant_url(collection: str) -> str:
    for url in ("http://127.0.0.1:6333", "http://localhost:6333"):
        try:
            with urllib.request.urlopen(f"{url}/collections", timeout=1.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            continue
        collections = payload.get("result", {}).get("collections", [])
        names = {str(item.get("name", "")).strip() for item in collections if isinstance(item, dict)}
        if collection in names:
            return url
    return ""


def _local_qdrant_collection_dirs(db_path: str, collection: str) -> List[Path]:
    if str(db_path or "").strip() == ":memory:":
        return []
    try:
        candidate_path = Path(db_path).resolve()
        return [
            candidate_path / "collections" / collection,
            candidate_path / "collection" / collection,
        ]
    except OSError:
        return []


def local_qdrant_collection_exists(db_path: str, collection: str) -> bool:
    for collection_dir in _local_qdrant_collection_dirs(db_path, collection):
        if not collection_dir.exists():
            continue
        sqlite_file = collection_dir / "storage.sqlite"
        if sqlite_file.exists() and sqlite_file.stat().st_size <= 0:
            continue
        return True
    return False


def resolve_local_qdrant_path(db_path: str, collection: str) -> str:
    if str(db_path or "").strip() == ":memory:":
        return ":memory:"
    candidates = []
    if db_path:
        candidates.append(Path(db_path))
    project_root = PIPELINE_ROOT
    workspace_root = project_root.parent
    temp_root = os.getenv("TEMP") or os.getenv("TMP") or ""
    candidates.extend(
        [
            project_root / "qdrant_storage",
            project_root / "rag_chunks_store" / "qdrant_local",
            project_root / "rag_chunks_store" / "qdrant_storage",
            workspace_root / "qdrant_storage",
            workspace_root / "rag_chunks_store" / "qdrant_local",
            workspace_root / "rag_chunks_store" / "qdrant_storage",
        ]
    )
    if temp_root:
        candidates.append(Path(temp_root) / "rag2_qdrant_local")
    seen = set()
    for candidate in candidates:
        try:
            candidate_path = Path(candidate).resolve()
        except OSError:
            continue
        key = str(candidate_path).lower()
        if key in seen:
            continue
        seen.add(key)
        if local_qdrant_collection_exists(str(candidate_path), collection):
            return str(candidate_path)
    try:
        return str(Path(db_path))
    except TypeError:
        return ""
        return False


def inspect_ingest_artifacts() -> Dict[str, Any]:
    project_root = PIPELINE_ROOT
    workspace_root = project_root.parent
    candidates = [
        project_root / "rag_chunks_store",
        workspace_root / "rag_chunks_store",
        project_root / "output",
        workspace_root / "output",
    ]
    best: Dict[str, Any] = {"dir": "", "chunk_files": 0, "embedded_files": 0}
    seen = set()
    for candidate in candidates:
        candidate_path = candidate.resolve()
        key = str(candidate_path).lower()
        if key in seen or not candidate_path.exists() or not candidate_path.is_dir():
            continue
        seen.add(key)
        chunk_files = len(list(candidate_path.rglob("*.chunks.json")))
        embedded_files = len(list(candidate_path.rglob("*.chunks.embedded.json")))
        if embedded_files > best["embedded_files"] or (
            embedded_files == best["embedded_files"] and chunk_files > best["chunk_files"]
        ):
            best = {
                "dir": str(candidate_path),
                "chunk_files": chunk_files,
                "embedded_files": embedded_files,
            }
    return best


def build_qdrant_unavailable_message(collection: str, url: str, db_path: str, error: BaseException) -> str:
    lines = [
        f"Qdrant server is not reachable: {url}",
        f"Original error: {error}",
    ]
    local_path = resolve_local_qdrant_path(db_path, collection)
    if local_qdrant_collection_exists(local_path, collection):
        lines.append(f"Local fallback is available at: {local_path}")
        return "\n".join(lines)

    lines.append(f"No local fallback collection was found under: {local_path}")
    artifacts = inspect_ingest_artifacts()
    artifact_dir = str(artifacts.get("dir") or "").strip()
    embedded_files = int(artifacts.get("embedded_files") or 0)
    script_dir = PIPELINE_ROOT
    if artifact_dir and embedded_files > 0:
        lines.append(f"Found {embedded_files} embedded chunk files under: {artifact_dir}")
        lines.append("Start Qdrant and sync the embedded chunks before searching:")
        lines.append(f'  docker compose -f "{script_dir / "docker-compose.qdrant.yml"}" up -d')
        lines.append(f'  powershell -ExecutionPolicy Bypass -File "{script_dir / "start_rag.ps1"}" sync')
    else:
        lines.append("No embedded chunk artifacts were found. Run ingest first:")
        lines.append(f'  powershell -ExecutionPolicy Bypass -File "{script_dir / "start_rag.ps1"}" ingest')
    return "\n".join(lines)


def build_missing_collection_message(collection: str, url: str, db_path: str) -> str:
    lines = [f"Qdrant collection does not exist: {collection}"]
    if url:
        lines.append(f"Connected Qdrant server: {url}")
    else:
        lines.append(f"Resolved local db_path: {db_path}")

    artifacts = inspect_ingest_artifacts()
    artifact_dir = str(artifacts.get("dir") or "").strip()
    chunk_files = int(artifacts.get("chunk_files") or 0)
    embedded_files = int(artifacts.get("embedded_files") or 0)
    script_dir = PIPELINE_ROOT
    python_exe = script_dir.parent / ".venv" / "Scripts" / "python.exe"
    embed_module = "rag_pipeline.ingest.embedding_qdrant"

    if artifact_dir and embedded_files > 0:
        lines.append(f"Found {embedded_files} embedded chunk files under: {artifact_dir}")
        lines.append("Sync them into Qdrant first, for example:")
        lines.append(
            f'  "{python_exe}" -m {embed_module} --input-path "{artifact_dir}" '
            f'--output-dir "{artifact_dir}" --url "http://127.0.0.1:6333" '
            f'--collection "{collection}" --reupsert-existing --no-write-json'
        )
    elif artifact_dir and chunk_files > 0:
        lines.append(f"Found {chunk_files} chunk files under: {artifact_dir}")
        lines.append("Ingest them before searching, for example:")
        lines.append(
            f'  powershell -ExecutionPolicy Bypass -File "{script_dir / "start_rag.ps1"}" ingest'
        )
    else:
        lines.append("No existing chunk artifacts were found. Run the ingest pipeline first.")

    return "\n".join(lines)


def is_qdrant_local_lock_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "already accessed by another instance of qdrant client" in message
        or "alreadylocked" in message
        or "storage folder" in message and "already accessed" in message
    )


def is_qdrant_remote_connection_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in [
            "connection refused",
            "connecterror",
            "winerror 10061",
            "actively refused",
            "failed to connect",
            "connection error",
            "connection aborted",
            "connection reset",
            "timed out",
            "timeout",
            "max retries exceeded",
        ]
    )


def build_qdrant_lock_message(db_path: str) -> str:
    return (
        f"Local Qdrant storage is locked: {db_path}\n"
        "This happens when another Python process has opened the same qdrant_local folder. "
        "Qdrant local mode only supports one active client per storage folder.\n"
        "Close the other running search/embedding process, or use Qdrant server mode for concurrent access "
        "by setting QDRANT_URL / --url."
    )


def preflight_runtime_checks(
    model_path: str,
    device: str,
    bge_m3_model_path: str = "",
    require_bge_m3: bool = False,
) -> None:
    issues = collect_runtime_dependency_issues(
        model_name_or_path=model_path,
        device=device,
        require_qdrant=True,
        bge_m3_model_path=bge_m3_model_path,
        require_bge_m3=require_bge_m3,
    )
    if issues:
        raise RuntimeError("Search preflight check failed:\n" + "\n".join(f"- {i}" for i in issues))


def get_qwen_embedder(
    model_path: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    max_length: int,
    keep_loaded: bool,
) -> QwenLocalEmbeddingModel:
    if not keep_loaded:
        return QwenLocalEmbeddingModel(
            model_name_or_path=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            max_length=max_length,
        )

    key = (
        str(model_path),
        str(device),
        str(dtype),
        str(attn_implementation),
        int(max_length),
    )
    embedder = _QWEN_EMBEDDER_CACHE.get(key)
    if embedder is None:
        embedder = QwenLocalEmbeddingModel(
            model_name_or_path=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            max_length=max_length,
        )
        _QWEN_EMBEDDER_CACHE[key] = embedder
    return embedder


def get_bge_m3_helper(
    model_path: str,
    device: str,
    batch_size: int,
    query_max_length: int,
    passage_max_length: int,
    use_fp16: bool,
    keep_loaded: bool,
) -> Optional[BgeM3IngestHelper]:
    resolved_model_path = str(model_path or "").strip()
    if not resolved_model_path:
        return None
    if not keep_loaded:
        return BgeM3IngestHelper(
            model_path=resolved_model_path,
            device=device,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            use_fp16=use_fp16,
        )

    key = (
        resolved_model_path,
        str(device or ""),
        int(batch_size or DEFAULT_BGE_M3_BATCH_SIZE),
        int(query_max_length or DEFAULT_BGE_M3_QUERY_MAX_LENGTH),
        int(passage_max_length or DEFAULT_BGE_M3_PASSAGE_MAX_LENGTH),
        bool(use_fp16),
    )
    helper = _BGE_M3_HELPER_CACHE.get(key)
    if helper is None:
        helper = BgeM3IngestHelper(
            model_path=resolved_model_path,
            device=device,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            use_fp16=use_fp16,
        )
        _BGE_M3_HELPER_CACHE[key] = helper
    return helper


def extract_collection_dense_vector_names(collection_info: Any) -> set[str]:
    vectors = getattr(getattr(collection_info, "config", None), "params", None)
    vectors = getattr(vectors, "vectors", None)
    if isinstance(vectors, dict):
        return {str(name) for name in vectors.keys()}
    if vectors is not None:
        return {DEFAULT_DENSE_VECTOR_NAME}
    return set()


def extract_collection_sparse_vector_names(collection_info: Any) -> set[str]:
    sparse_vectors = getattr(getattr(collection_info, "config", None), "params", None)
    sparse_vectors = getattr(sparse_vectors, "sparse_vectors", None)
    if isinstance(sparse_vectors, dict):
        return {str(name) for name in sparse_vectors.keys()}
    return set()


def build_qdrant_sparse_vectors(qmodels: Any, sparse_payloads: Sequence[Dict[str, Any]]) -> List[Any]:
    vectors: List[Any] = []
    for sparse_payload in sparse_payloads:
        indices = [int(value) for value in sparse_payload.get("indices", [])]
        values = [float(value) for value in sparse_payload.get("values", [])]
        if not indices or not values:
            continue
        vectors.append(qmodels.SparseVector(indices=indices, values=values))
    return vectors


def build_filter(args, qmodels, plan: Optional[QueryPlan] = None):
    conditions = []
    if args.source_file:
        conditions.append(qmodels.FieldCondition(key="source_file", match=qmodels.MatchValue(value=args.source_file)))
    if args.doc_title:
        conditions.append(qmodels.FieldCondition(key="doc_title", match=qmodels.MatchValue(value=args.doc_title)))
    if args.chunk_uid:
        conditions.append(qmodels.FieldCondition(key="chunk_uid", match=qmodels.MatchValue(value=args.chunk_uid)))
    if args.chunk_type:
        conditions.append(qmodels.FieldCondition(key="chunk_type", match=qmodels.MatchValue(value=args.chunk_type)))
    if getattr(args, "chunk_level", ""):
        conditions.append(qmodels.FieldCondition(key="chunk_level", match=qmodels.MatchValue(value=args.chunk_level)))
    if plan and getattr(args, "planner_hard_filters", False):
        for key, values in (plan.filter_hints or {}).items():
            if key not in _HARD_FILTER_HINT_KEYS:
                continue
            hint_filter = match_any_filter(qmodels, key, values)
            if hint_filter:
                conditions.append(hint_filter)
    return qmodels.Filter(must=conditions) if conditions else None


def combine_filters(qmodels, *filters):
    must_conditions = []
    for filter_obj in filters:
        if not filter_obj:
            continue
        if getattr(filter_obj, "must", None):
            must_conditions.extend(filter_obj.must)
        else:
            must_conditions.append(filter_obj)
    return qmodels.Filter(must=must_conditions) if must_conditions else None


def match_any_filter(qmodels, key: str, values: Sequence[str]):
    cleaned_values = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned_values:
        return None
    if hasattr(qmodels, "MatchAny"):
        return qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=cleaned_values))
    should = [qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value)) for value in cleaned_values]
    return qmodels.Filter(should=should)


def point_to_candidate(point: Any, source_name: str, preview_chars: int) -> SearchCandidate:
    payload = point.payload or {}
    header_path = payload.get("header_path", [])
    if not isinstance(header_path, list):
        header_path = [str(header_path)] if header_path else []
    text = cleanup_retrieval_text(str(payload.get("text", "") or payload.get("retrieval_text", "") or ""))
    return SearchCandidate(
        id=str(point.id),
        chunk_uid=str(payload.get("chunk_uid", point.id)),
        chunk_level=str(payload.get("chunk_level", "child") or "child"),
        parent_chunk_uid=str(payload.get("parent_chunk_uid", "") or ""),
        source_name=source_name,
        semantic_score=float(point.score),
        doc_title=str(payload.get("doc_title", "") or ""),
        source_file=str(payload.get("source_file", "") or ""),
        section_title=str(payload.get("section_title", "") or ""),
        header_path=header_path,
        chunk_type=str(payload.get("chunk_type", "") or ""),
        quality_score=float(payload.get("quality_score", 0.0) or 0.0),
        text_preview=text[:preview_chars],
        text=text,
        payload=payload,
        matched_queries=[],
        score_breakdown={},
    )


def unique_key(candidate: SearchCandidate) -> str:
    return "|".join([
        candidate.chunk_uid,
        candidate.chunk_level,
        normalize_text(candidate.doc_title),
        normalize_text(candidate.section_title),
    ])

def dense_search(
    client: Any,
    collection: str,
    query_variants: Sequence[str],
    query_vectors: Sequence[Any],
    query_filter: Any,
    search_params: Any,
    score_threshold: Optional[float],
    limit: int,
    preview_chars: int,
    source_name: str,
    score_key: str,
    vector_name: str = DEFAULT_DENSE_VECTOR_NAME,
) -> List[SearchCandidate]:
    merged: Dict[str, SearchCandidate] = {}
    for variant, vector in zip(query_variants, query_vectors):
        response = client.query_points(
            collection_name=collection,
            query=vector,
            using=vector_name,
            query_filter=query_filter,
            search_params=search_params,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )
        for rank, point in enumerate(response.points, start=1):
            candidate = point_to_candidate(point, source_name=source_name, preview_chars=preview_chars)
            key = unique_key(candidate)
            if key not in merged:
                candidate.matched_queries = [variant]
                candidate.score_breakdown = {
                    score_key: float(candidate.semantic_score),
                    "fusion_rrf": weighted_rrf(rank, source_name, score_key),
                }
                merged[key] = candidate
            else:
                existing = merged[key]
                existing.semantic_score = max(existing.semantic_score, candidate.semantic_score)
                if variant not in existing.matched_queries:
                    existing.matched_queries.append(variant)
                existing.score_breakdown[score_key] = max(
                    existing.score_breakdown.get(score_key, 0.0),
                    candidate.semantic_score,
                )
                existing.score_breakdown["fusion_rrf"] = existing.score_breakdown.get("fusion_rrf", 0.0) + weighted_rrf(rank, source_name, score_key)
    return list(merged.values())


def sparse_search(
    client: Any,
    collection: str,
    query_variants: Sequence[str],
    query_sparse_vectors: Sequence[Any],
    query_filter: Any,
    limit: int,
    preview_chars: int,
    source_name: str,
    score_key: str,
    vector_name: str = DEFAULT_SPARSE_VECTOR_NAME,
) -> List[SearchCandidate]:
    merged: Dict[str, SearchCandidate] = {}
    for variant, sparse_vector in zip(query_variants, query_sparse_vectors):
        response = client.query_points(
            collection_name=collection,
            query=sparse_vector,
            using=vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        for rank, point in enumerate(response.points, start=1):
            candidate = point_to_candidate(point, source_name=source_name, preview_chars=preview_chars)
            key = unique_key(candidate)
            if key not in merged:
                candidate.matched_queries = [variant]
                candidate.score_breakdown = {
                    score_key: float(candidate.semantic_score),
                    "fusion_rrf": weighted_rrf(rank, source_name, score_key),
                }
                merged[key] = candidate
            else:
                existing = merged[key]
                existing.semantic_score = max(existing.semantic_score, candidate.semantic_score)
                if variant not in existing.matched_queries:
                    existing.matched_queries.append(variant)
                existing.score_breakdown[score_key] = max(
                    existing.score_breakdown.get(score_key, 0.0),
                    candidate.semantic_score,
                )
                existing.score_breakdown["fusion_rrf"] = existing.score_breakdown.get("fusion_rrf", 0.0) + weighted_rrf(rank, source_name, score_key)
    return list(merged.values())


def attach_parent_scores(
    child_candidates: Sequence[SearchCandidate],
    parent_candidates: Sequence[SearchCandidate],
) -> None:
    parent_scores = {
        parent.chunk_uid: parent.score_breakdown.get("final_score", parent.semantic_score)
        for parent in parent_candidates
    }
    for candidate in child_candidates:
        parent_uid = candidate.parent_chunk_uid
        if parent_uid and parent_uid in parent_scores:
            candidate.score_breakdown["parent_recall_score"] = float(parent_scores[parent_uid])


def needs_flat_child_fallback(query: str, child_candidates: Sequence[SearchCandidate]) -> bool:
    if not child_candidates:
        return True
    top_candidates = sorted(child_candidates, key=lambda item: item.semantic_score, reverse=True)[:3]
    best_support = max(compute_body_support_score(query, candidate.payload) for candidate in top_candidates)
    return best_support < 0.10


def hierarchical_dense_search(
    client: Any,
    qmodels: Any,
    collection: str,
    query_variants: Sequence[str],
    query_vectors: Sequence[Any],
    query_sparse_vectors: Sequence[Any],
    base_filter: Any,
    search_params: Any,
    score_threshold: Optional[float],
    limit: int,
    preview_chars: int,
    source_name: str,
    score_key: str,
    dense_vector_name: str = DEFAULT_DENSE_VECTOR_NAME,
    sparse_source_name: str = "lexical_sparse",
    sparse_score_key: str = "lexical_sparse",
    sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME,
) -> tuple[List[SearchCandidate], List[SearchCandidate]]:
    parent_filter = combine_filters(
        qmodels,
        base_filter,
        qmodels.Filter(must=[qmodels.FieldCondition(key="chunk_level", match=qmodels.MatchValue(value="parent"))]),
    )
    parent_dense_candidates = dense_search(
        client=client,
        collection=collection,
        query_variants=query_variants,
        query_vectors=query_vectors,
        query_filter=parent_filter,
        search_params=search_params,
        score_threshold=score_threshold,
        limit=limit,
        preview_chars=preview_chars,
        source_name=source_name,
        score_key=score_key,
        vector_name=dense_vector_name,
    )
    parent_sparse_candidates = []
    if query_sparse_vectors:
        parent_sparse_candidates = sparse_search(
            client=client,
            collection=collection,
            query_variants=query_variants,
            query_sparse_vectors=query_sparse_vectors,
            query_filter=parent_filter,
            limit=limit,
            preview_chars=preview_chars,
            source_name=sparse_source_name,
            score_key=sparse_score_key,
            vector_name=sparse_vector_name,
        )
    parent_candidates = merge_candidates(parent_dense_candidates, parent_sparse_candidates)
    if not parent_candidates:
        return [], []

    reranked_parents = rerank_candidates(
        query=query_variants[0] if query_variants else "",
        candidates=parent_candidates,
        top_k=min(len(parent_candidates), max(8, math.ceil(limit / 2))),
        max_per_document=max(4, DEFAULT_MAX_PER_DOCUMENT * 2),
        use_external_rerank=False,
    )
    parent_uids = [candidate.chunk_uid for candidate in reranked_parents if candidate.chunk_uid]
    child_scope = match_any_filter(qmodels, "parent_chunk_uid", parent_uids)
    child_filter = combine_filters(
        qmodels,
        base_filter,
        qmodels.Filter(must=[qmodels.FieldCondition(key="chunk_level", match=qmodels.MatchValue(value="child"))]),
        child_scope,
    )
    child_dense_candidates = dense_search(
        client=client,
        collection=collection,
        query_variants=query_variants,
        query_vectors=query_vectors,
        query_filter=child_filter,
        search_params=search_params,
        score_threshold=score_threshold,
        limit=max(limit, len(parent_uids) * 4),
        preview_chars=preview_chars,
        source_name=source_name,
        score_key=score_key,
        vector_name=dense_vector_name,
    )
    child_sparse_candidates = []
    if query_sparse_vectors:
        child_sparse_candidates = sparse_search(
            client=client,
            collection=collection,
            query_variants=query_variants,
            query_sparse_vectors=query_sparse_vectors,
            query_filter=child_filter,
            limit=max(limit, len(parent_uids) * 4),
            preview_chars=preview_chars,
            source_name=sparse_source_name,
            score_key=sparse_score_key,
            vector_name=sparse_vector_name,
        )
    child_candidates = merge_candidates(child_dense_candidates, child_sparse_candidates)
    attach_parent_scores(child_candidates, reranked_parents)
    return reranked_parents, child_candidates


def flat_child_search(
    client: Any,
    qmodels: Any,
    collection: str,
    query_variants: Sequence[str],
    query_vectors: Sequence[Any],
    query_sparse_vectors: Sequence[Any],
    base_filter: Any,
    search_params: Any,
    score_threshold: Optional[float],
    limit: int,
    preview_chars: int,
    dense_source_name: str = "qwen_dense_flat_child",
    dense_score_key: str = "qwen_dense_flat_child",
    dense_vector_name: str = DEFAULT_DENSE_VECTOR_NAME,
    sparse_source_name: str = "lexical_sparse_flat_child",
    sparse_score_key: str = "lexical_sparse_flat_child",
    sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME,
) -> List[SearchCandidate]:
    flat_child_filter = combine_filters(
        qmodels,
        base_filter,
        qmodels.Filter(must=[qmodels.FieldCondition(key="chunk_level", match=qmodels.MatchValue(value="child"))]),
    )
    flat_child_dense = dense_search(
        client=client,
        collection=collection,
        query_variants=query_variants,
        query_vectors=query_vectors,
        query_filter=flat_child_filter,
        search_params=search_params,
        score_threshold=score_threshold,
        limit=limit,
        preview_chars=preview_chars,
        source_name=dense_source_name,
        score_key=dense_score_key,
        vector_name=dense_vector_name,
    )
    flat_child_sparse = []
    if query_sparse_vectors:
        flat_child_sparse = sparse_search(
            client=client,
            collection=collection,
            query_variants=query_variants,
            query_sparse_vectors=query_sparse_vectors,
            query_filter=flat_child_filter,
            limit=limit,
            preview_chars=preview_chars,
            source_name=sparse_source_name,
            score_key=sparse_score_key,
            vector_name=sparse_vector_name,
        )
    return merge_candidates(flat_child_dense, flat_child_sparse)


def merge_candidates(*candidate_groups: Sequence[SearchCandidate]) -> List[SearchCandidate]:
    merged: Dict[str, SearchCandidate] = {}
    for group in candidate_groups:
        for candidate in group:
            key = unique_key(candidate)
            if key not in merged:
                merged[key] = candidate
                continue
            existing = merged[key]
            existing.semantic_score = max(existing.semantic_score, candidate.semantic_score)
            for q in candidate.matched_queries:
                if q not in existing.matched_queries:
                    existing.matched_queries.append(q)
            for score_name, score_value in candidate.score_breakdown.items():
                if score_name == "fusion_rrf":
                    existing.score_breakdown[score_name] = existing.score_breakdown.get(score_name, 0.0) + score_value
                else:
                    existing.score_breakdown[score_name] = max(existing.score_breakdown.get(score_name, 0.0), score_value)
            if candidate.source_name not in existing.source_name:
                existing.source_name = existing.source_name + "+" + candidate.source_name
    return list(merged.values())


def build_rerank_document(candidate: SearchCandidate, max_chars: int) -> str:
    header = " > ".join(candidate.header_path) if candidate.header_path else ""
    body = str(candidate.payload.get("embedding_text", "") or "").strip()
    if not body:
        body = candidate.text or candidate.text_preview or ""

    text = "\n".join(
        part for part in [
            f"doc_title: {candidate.doc_title}" if candidate.doc_title else "",
            f"section_title: {candidate.section_title}" if candidate.section_title else "",
            f"header_path: {header}" if header else "",
            f"source_file: {candidate.source_file}" if candidate.source_file else "",
            f"chunk_level: {candidate.chunk_level}" if candidate.chunk_level else "",
            f"chunk_type: {candidate.chunk_type}" if candidate.chunk_type else "",
            f"text: {body}" if body else "",
        ] if part
    ).strip()

    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def candidate_qwen_dense_score(candidate: SearchCandidate) -> float:
    return max(
        float(candidate.score_breakdown.get("qwen_dense", 0.0) or 0.0),
        float(candidate.score_breakdown.get("qwen_dense_flat_child", 0.0) or 0.0),
    )


def candidate_bge_dense_score(candidate: SearchCandidate) -> float:
    return max(
        float(candidate.score_breakdown.get("bge_dense", 0.0) or 0.0),
        float(candidate.score_breakdown.get("bge_dense_flat_child", 0.0) or 0.0),
    )


def candidate_sparse_score(candidate: SearchCandidate, *keys: str) -> float:
    return max(float(candidate.score_breakdown.get(key, 0.0) or 0.0) for key in keys)


def post_external_json(api_url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("External API calls require the optional 'requests' package.") from exc

    with requests.Session() as session:
        session.trust_env = DEFAULT_EXTERNAL_API_TRUST_ENV
        response = session.post(api_url, headers=headers, json=payload, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if detail:
                raise RuntimeError(f"External API request failed with HTTP {response.status_code}: {detail}") from exc
            raise
        try:
            return response.json()
        except ValueError as exc:
            detail = response.text[:800].strip()
            raise RuntimeError(f"External API returned non-JSON content: {detail}") from exc


def call_external_rerank_api(
    query: str,
    documents: Sequence[str],
    provider: str,
    api_url: str,
    api_key: str,
    model: str,
    top_n: int,
    timeout: float,
) -> List[Dict[str, Any]]:
    if not api_key:
        raise RuntimeError("External rerank API key is empty.")

    provider = normalize_text(provider)

    if provider == "dashscope" or "dashscope.aliyuncs.com/api/v1/services/rerank" in str(api_url).lower():
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "input": {
                "query": query,
                "documents": list(documents),
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": False,
            },
        }
        data = post_external_json(api_url=api_url, headers=headers, payload=payload, timeout=timeout)
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        results = output.get("results") if isinstance(output.get("results"), list) else data.get("results", [])
        return results if isinstance(results, list) else []

    if provider == "cohere":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
            "max_tokens_per_doc": 4096,
        }
        data = post_external_json(api_url=api_url, headers=headers, payload=payload, timeout=timeout)
        return data.get("results", [])

    if provider == "jina":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
            "return_documents": False,
        }
        data = post_external_json(api_url=api_url, headers=headers, payload=payload, timeout=timeout)
        return data.get("results", [])

    raise RuntimeError(f"Unsupported rerank provider: {provider}")


def add_external_rerank_scores(
    query: str,
    candidates: Sequence[SearchCandidate],
    provider: str,
    api_url: str,
    api_key: str,
    model: str,
    max_docs: int,
    top_n: int,
    max_chars_per_doc: int,
    timeout: float,
) -> bool:
    if not candidates or not api_key:
        return False

    rerank_candidates = sorted(
        list(candidates),
        key=lambda c: (
            candidate_qwen_dense_score(c),
            candidate_bge_dense_score(c),
            candidate_sparse_score(c, "lexical_sparse", "lexical_sparse_flat_child", "bge_sparse", "bge_sparse_flat_child"),
            c.score_breakdown.get("fusion_rrf", 0.0),
            c.quality_score,
        ),
        reverse=True,
    )[: max(1, min(len(candidates), max_docs))]

    documents = [build_rerank_document(c, max_chars=max_chars_per_doc) for c in rerank_candidates]
    try:
        results = call_external_rerank_api(
            query=query,
            documents=documents,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            model=model,
            top_n=max(1, min(top_n, len(documents))),
            timeout=timeout,
        )
    except Exception as exc:
        for candidate in rerank_candidates:
            candidate.score_breakdown["api_rerank_error"] = str(exc)[:300]
        return False

    assigned = False
    for rank, item in enumerate(results, start=1):
        idx = int(item["index"])
        if idx < 0 or idx >= len(rerank_candidates):
            continue
        score = float(item.get("relevance_score", 0.0) or 0.0)
        candidate = rerank_candidates[idx]
        candidate.score_breakdown["api_rerank_score"] = score
        candidate.score_breakdown["api_rerank_rank"] = float(rank)
        assigned = True
    return assigned


def get_local_cross_encoder(model_path: str):
    resolved = str(model_path or "").strip()
    if not resolved:
        return None
    if resolved in _LOCAL_RERANKER_CACHE:
        return _LOCAL_RERANKER_CACHE[resolved]
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RuntimeError("Local rerank requires sentence-transformers and a local cross-encoder model path.") from exc
    model = CrossEncoder(resolved, trust_remote_code=True)
    _LOCAL_RERANKER_CACHE[resolved] = model
    return model


def add_local_rerank_scores(
    query: str,
    candidates: Sequence[SearchCandidate],
    model_path: str,
    max_docs: int,
    max_chars_per_doc: int,
    batch_size: int,
) -> None:
    if not candidates or not str(model_path or "").strip():
        return
    reranker = get_local_cross_encoder(model_path)
    if reranker is None:
        return
    rerank_candidates = sorted(
        list(candidates),
        key=lambda c: (
            c.score_breakdown.get("final_score", 0.0),
            candidate_qwen_dense_score(c),
            candidate_bge_dense_score(c),
            candidate_sparse_score(c, "lexical_sparse", "lexical_sparse_flat_child", "bge_sparse", "bge_sparse_flat_child"),
            c.score_breakdown.get("fusion_rrf", 0.0),
            c.quality_score,
        ),
        reverse=True,
    )[: max(1, min(len(candidates), max_docs))]
    pairs = [(query, build_rerank_document(candidate, max_chars=max_chars_per_doc)) for candidate in rerank_candidates]
    scores = reranker.predict(pairs, batch_size=max(1, int(batch_size)))
    ranked_scores = sorted(enumerate([float(score) for score in scores]), key=lambda item: item[1], reverse=True)
    for rank, (idx, score) in enumerate(ranked_scores, start=1):
        candidate = rerank_candidates[idx]
        candidate.score_breakdown["local_rerank_score"] = score
        candidate.score_breakdown["local_rerank_rank"] = float(rank)


def assign_evidence_group(plan: QueryPlan, candidate: SearchCandidate, quote: str) -> str:
    if not plan.sub_queries or len(plan.sub_queries) <= 1:
        return "primary"
    best_group = "primary"
    best_overlap = -1
    target_text = normalize_text(" ".join([candidate.section_title, quote]))
    for idx, sub_query in enumerate(plan.sub_queries[1:], start=1):
        terms = build_query_terms(sub_query)
        overlap = sum(1 for term in terms if normalize_text(term) and normalize_text(term) in target_text)
        if overlap > best_overlap:
            best_overlap = overlap
            best_group = f"sub_question_{idx}"
    return best_group


def looks_numeric_or_temporal_query(query: str) -> bool:
    return bool(_NUMERIC_QUERY_RE.search(str(query or "")) or _TIME_TOKEN_RE.search(str(query or "")))


def detect_evidence_conflicts(query: str, evidence_items: Sequence[EvidenceItem]) -> List[Dict[str, Any]]:
    if len(evidence_items) < 2 or not looks_numeric_or_temporal_query(query):
        return []

    token_map: Dict[str, List[str]] = defaultdict(list)
    for item in evidence_items[:5]:
        source_label = item.doc_title or Path(item.source_file).name or item.chunk_uid
        for token in unique_preserve(_NUMERIC_TOKEN_RE.findall(item.quote)):
            token_map[token].append(source_label)

    if len(token_map) < 2:
        return []

    return [
        {
            "type": "numeric_or_temporal_variation",
            "message": "Selected evidence contains multiple date/number values. Verify the exact condition before finalizing the answer.",
            "values": [
                {
                    "value": token,
                    "sources": unique_preserve(sources),
                }
                for token, sources in sorted(token_map.items(), key=lambda item: item[0])[:6]
            ],
        }
    ]


def get_task_required_facets(task_type: str) -> List[str]:
    return list(_TASK_FACET_GROUPS.get(task_type, {}).keys())


def get_task_facets_from_text(text: str, task_type: str) -> set[str]:
    lowered = normalize_text(text)
    if not lowered:
        return set()
    facets = set()
    for facet_name, tokens in _TASK_FACET_GROUPS.get(task_type, {}).items():
        if any(token in lowered for token in tokens):
            facets.add(facet_name)
    return facets


def build_topic_anchor_terms(plan: QueryPlan) -> List[str]:
    seed_terms = plan.entity_terms + extract_subject_terms(plan.original_query) + plan.theme_terms + build_query_terms(plan.original_query)
    candidates: List[str] = []
    for item in seed_terms:
        candidates.append(item)
        candidates.extend(build_query_terms(item))
    anchors: List[str] = []
    for item in candidates:
        normalized = normalize_text(item)
        if len(normalized) < 2 or normalized in _GENERIC_QUERY_STOP_TOKENS:
            continue
        anchors.append(normalized)
    if not anchors:
        anchors = [term for term in build_query_terms(plan.original_query) if len(term) >= 2][:4]
    return unique_preserve(anchors, max_items=6)


def build_candidate_analysis_text(candidate: SearchCandidate, quote: str = "") -> str:
    parts = [
        candidate.doc_title,
        candidate.section_title,
        " ".join(candidate.header_path),
        quote,
        candidate.payload.get("retrieval_text", ""),
        candidate.text_preview,
        candidate.text,
    ]
    return normalize_text(" ".join(str(part or "") for part in parts if str(part or "").strip()))


def compute_task_alignment_breakdown(plan: Optional[QueryPlan], candidate: SearchCandidate, quote: str = "") -> Dict[str, Any]:
    if not plan:
        return {
            "task_alignment_score": 0.0,
            "facet_coverage_score": 0.0,
            "doctype_fitness_score": 0.0,
            "offtopic_penalty": 0.0,
            "anchor_ratio": 0.0,
            "detected_facets": [],
        }

    payload = candidate.payload or {}
    doc_type = normalize_text(payload.get("doc_type", ""))
    chunk_type = normalize_text(payload.get("chunk_type", ""))
    text = build_candidate_analysis_text(candidate, quote=quote)
    required_facets = set(get_task_required_facets(plan.task_type))
    detected_facets = get_task_facets_from_text(text, plan.task_type)
    positive_doc_types = {normalize_text(value) for value in (plan.filter_hints or {}).get("doc_type", [])}
    positive_chunk_types = {normalize_text(value) for value in (plan.filter_hints or {}).get("chunk_type", [])}
    negative_doc_types = {normalize_text(value) for value in (plan.filter_hints or {}).get("negative_doc_type", [])}
    negative_chunk_types = {normalize_text(value) for value in (plan.filter_hints or {}).get("negative_chunk_type", [])}

    doctype_fitness_score = 0.0
    if doc_type and doc_type in positive_doc_types:
        doctype_fitness_score += 0.55
    if chunk_type and chunk_type in positive_chunk_types:
        doctype_fitness_score += 0.35
    if plan.task_type in {"comparison", "market"} and chunk_type in {"table", "list"}:
        doctype_fitness_score += 0.15
    if plan.task_type == "definition" and chunk_type in {"summary", "heading", "title"}:
        doctype_fitness_score += 0.12
    doctype_fitness_score = min(1.0, doctype_fitness_score)

    anchors = build_topic_anchor_terms(plan)
    anchor_hits = sum(1 for term in anchors if term in text)
    anchor_ratio = min(1.0, anchor_hits / max(len(anchors), 1))

    facet_coverage_score = 0.0
    if required_facets:
        facet_coverage_score = len(detected_facets & required_facets) / max(len(required_facets), 1)

    offtopic_penalty = 0.0
    if doc_type and doc_type in negative_doc_types:
        offtopic_penalty += 0.50
    if chunk_type and chunk_type in negative_chunk_types:
        offtopic_penalty += 0.25
    negative_hits = sum(1 for token in _TASK_NEGATIVE_TERMS.get(plan.task_type, []) if token in text)
    if negative_hits:
        offtopic_penalty += min(0.60, 0.12 * negative_hits)
    if required_facets and not detected_facets and plan.task_type not in {"definition", "fact"}:
        offtopic_penalty += 0.18

    task_alignment_score = max(
        0.0,
        min(
            1.0,
            (0.45 * facet_coverage_score)
            + (0.30 * doctype_fitness_score)
            + (0.25 * anchor_ratio)
            - (0.35 * offtopic_penalty),
        ),
    )

    return {
        "task_alignment_score": float(task_alignment_score),
        "facet_coverage_score": float(facet_coverage_score),
        "doctype_fitness_score": float(doctype_fitness_score),
        "offtopic_penalty": float(offtopic_penalty),
        "anchor_ratio": float(anchor_ratio),
        "detected_facets": sorted(detected_facets),
    }


def score_plan_alignment(plan: QueryPlan, candidate: SearchCandidate) -> float:
    payload = candidate.payload or {}
    score = 0.0
    doc_type = normalize_text(payload.get("doc_type", ""))
    chunk_type = normalize_text(payload.get("chunk_type", ""))
    section_kind = normalize_text(payload.get("section_kind", ""))
    evidence_focus = {normalize_text(item) for item in plan.evidence_focus}
    filter_hints = {key: [normalize_text(v) for v in values] for key, values in (plan.filter_hints or {}).items()}
    task_alignment = compute_task_alignment_breakdown(plan, candidate)

    if doc_type and doc_type in filter_hints.get("doc_type", []):
        score += 0.18
    if chunk_type and chunk_type in filter_hints.get("chunk_type", []):
        score += 0.18
    if "policy_clause" in evidence_focus and section_kind in {"policy", "rule", "procedure"}:
        score += 0.12
    if "technical_fix" in evidence_focus and chunk_type in {"faq", "code", "example"}:
        score += 0.10
    if "structured_list" in evidence_focus and chunk_type in {"table", "list"}:
        score += 0.10

    score += 0.20 * task_alignment["facet_coverage_score"]
    score += 0.18 * task_alignment["doctype_fitness_score"]
    score += 0.12 * task_alignment["anchor_ratio"]
    score -= 0.22 * task_alignment["offtopic_penalty"]
    return max(0.0, min(1.0, score))


def extract_best_quote(query: str, plan: QueryPlan, candidate: SearchCandidate, max_chars: int = 240) -> str:
    text = cleanup_retrieval_text(str(candidate.payload.get("retrieval_text", "") or candidate.text or "").strip())
    if not text:
        return ""

    query_terms = build_body_match_terms(query)
    query_terms.extend(plan.theme_terms)
    query_terms.extend(build_topic_anchor_terms(plan))
    query_terms = unique_preserve(query_terms)
    sentences = split_sentences(text)
    if not sentences:
        return text[:max_chars]

    scored: List[tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences):
        cleaned_sentence = cleanup_retrieval_text(sentence)
        if not cleaned_sentence:
            continue
        lowered = normalize_text(cleaned_sentence)
        overlap = sum(1 for term in query_terms if term and term in lowered)
        overlap_ratio = overlap / max(len(query_terms), 1)
        sentence_score = overlap + overlap_ratio
        task_signal = quote_task_signal_score(plan, cleaned_sentence)
        sentence_score += 1.8 * task_signal
        if plan.task_type == "trend" and is_industry_thesis_query(plan):
            sentence_score -= 1.2 * quote_offtopic_penalty(plan, cleaned_sentence)
        if "logic" in lowered or "content type:" in lowered:
            sentence_score -= 0.8
        if re.search(r"第\s*\d+\s*页", cleaned_sentence):
            sentence_score -= 0.6
        if re.search(r"[A-Za-z]:\\", cleaned_sentence):
            sentence_score -= 0.6
        if count_cjk(cleaned_sentence) < 8 and len(cleaned_sentence) < 24:
            sentence_score -= 0.4
        if plan.time_terms and any(normalize_text(token) in lowered for token in plan.time_terms):
            sentence_score += 0.8
        if candidate.section_title and normalize_text(candidate.section_title) in lowered:
            sentence_score += 0.3
        scored.append((sentence_score, idx, cleaned_sentence))

    if not scored:
        return text[:max_chars]

    scored.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    _, _, best_sentence = scored[0]
    if len(best_sentence) < max_chars:
        extra_parts: List[str] = [best_sentence]
        for _, _, sentence in scored[1:4]:
            if sentence == best_sentence:
                continue
            if count_cjk(sentence) < 6 and len(sentence) < 24:
                continue
            if plan.task_type == "trend" and is_industry_thesis_query(plan):
                if quote_task_signal_score(plan, sentence) < 0.15:
                    continue
            merged = "；".join(extra_parts + [sentence])
            if len(merged) > max_chars:
                continue
            extra_parts.append(sentence)
            if len(extra_parts) >= 2:
                break
        best_sentence = "；".join(extra_parts)
    return cleanup_retrieval_text(best_sentence)[:max_chars]


def normalize_rank_signal(value: float, *, rrf: bool = False) -> float:
    value = max(0.0, float(value or 0.0))
    if rrf:
        return min(1.0, value * 60.0)
    if value <= 1.0:
        return value
    return value / (1.0 + value)


def rerank_candidates(
    query: str,
    candidates: Sequence[SearchCandidate],
    top_k: int,
    max_per_document: int,
    use_external_rerank: bool = False,
    plan: Optional[QueryPlan] = None,
) -> List[SearchCandidate]:
    enriched: List[Dict[str, Any]] = []
    for c in candidates:
        lexical_bonus = compute_lexical_bonus(query, c.payload)
        metadata_bonus = compute_metadata_bonus(c.payload | c.to_dict())
        query_match_bonus = min(0.10, 0.03 * len(c.matched_queries))
        body_support_score = compute_body_support_score(query, c.payload)
        plan_alignment = score_plan_alignment(plan, c) if plan else 0.0
        task_alignment = compute_task_alignment_breakdown(plan, c) if plan else compute_task_alignment_breakdown(None, c)
        qwen_dense = normalize_rank_signal(max(
            c.score_breakdown.get("qwen_dense", 0.0),
            c.score_breakdown.get("qwen_dense_flat_child", 0.0),
        ))
        bge_dense = normalize_rank_signal(max(
            c.score_breakdown.get("bge_dense", 0.0),
            c.score_breakdown.get("bge_dense_flat_child", 0.0),
        ))
        lexical_sparse_raw = max(
            c.score_breakdown.get("lexical_sparse", 0.0),
            c.score_breakdown.get("lexical_sparse_flat_child", 0.0),
        )
        lexical_sparse = normalize_rank_signal(math.log1p(max(0.0, lexical_sparse_raw)))
        bge_sparse_raw = max(
            c.score_breakdown.get("bge_sparse", 0.0),
            c.score_breakdown.get("bge_sparse_flat_child", 0.0),
        )
        bge_sparse = normalize_rank_signal(math.log1p(max(0.0, bge_sparse_raw)))
        fusion_rrf = normalize_rank_signal(c.score_breakdown.get("fusion_rrf", 0.0), rrf=True)
        api_rerank_score = normalize_rank_signal(c.score_breakdown.get("api_rerank_score", 0.0))
        local_rerank_score = normalize_rank_signal(c.score_breakdown.get("local_rerank_score", 0.0))
        rerank_score = max(api_rerank_score, local_rerank_score)
        parent_recall_score = c.score_breakdown.get("parent_recall_score", 0.0)

        if (use_external_rerank and "api_rerank_score" in c.score_breakdown) or "local_rerank_score" in c.score_breakdown:
            hybrid_base = (
                (1.00 * rerank_score)
                + (0.20 * fusion_rrf)
                + (0.08 * qwen_dense)
                + (0.06 * bge_dense)
                + (0.08 * lexical_sparse)
                + (0.10 * bge_sparse)
            )
        else:
            hybrid_base = (
                (0.95 * qwen_dense)
                + (0.70 * bge_dense)
                + (0.75 * lexical_sparse)
                + (0.95 * bge_sparse)
                + (1.20 * fusion_rrf)
            )

        final_score = hybrid_base + (0.35 * lexical_bonus) + (0.25 * metadata_bonus) + (0.15 * query_match_bonus)
        final_score += 0.40 * body_support_score
        final_score += 0.16 * plan_alignment
        final_score += 0.22 * task_alignment["task_alignment_score"]
        final_score += 0.14 * task_alignment["facet_coverage_score"]
        final_score += 0.12 * task_alignment["doctype_fitness_score"]
        final_score -= 0.22 * task_alignment["offtopic_penalty"]

        if c.chunk_level == "child":
            final_score += 0.20 * parent_recall_score
            if body_support_score < 0.08 and lexical_bonus >= 0.22:
                final_score -= 0.30
            elif body_support_score < 0.12 and lexical_bonus >= 0.18:
                final_score -= 0.16
        elif c.chunk_level == "parent":
            final_score += 0.05 * parent_recall_score

        if plan and plan.task_type == "market" and task_alignment["facet_coverage_score"] <= 0.0 and body_support_score < 0.15:
            final_score -= 0.18
        if plan and task_alignment["offtopic_penalty"] >= 0.60 and task_alignment["task_alignment_score"] < 0.20:
            final_score -= 0.25

        enriched.append(
            {
                "candidate": c,
                "final_score": final_score,
                "hybrid_base": hybrid_base,
                "qwen_dense": qwen_dense,
                "bge_dense": bge_dense,
                "lexical_sparse": lexical_sparse,
                "lexical_sparse_raw": lexical_sparse_raw,
                "bge_sparse": bge_sparse,
                "bge_sparse_raw": bge_sparse_raw,
                "lexical_bonus": lexical_bonus,
                "metadata_bonus": metadata_bonus,
                "query_match_bonus": query_match_bonus,
                "api_rerank_score": api_rerank_score,
                "local_rerank_score": local_rerank_score,
                "rerank_score": rerank_score,
                "parent_recall_score": parent_recall_score,
                "body_support_score": body_support_score,
                "plan_alignment": plan_alignment,
                "task_alignment_score": task_alignment["task_alignment_score"],
                "facet_coverage_score": task_alignment["facet_coverage_score"],
                "doctype_fitness_score": task_alignment["doctype_fitness_score"],
                "offtopic_penalty": task_alignment["offtopic_penalty"],
            }
        )

    enriched.sort(
        key=lambda x: (
            x["final_score"],
            x["task_alignment_score"],
            -x["offtopic_penalty"],
            x["api_rerank_score"],
            x["local_rerank_score"],
            x["candidate"].semantic_score,
            x["candidate"].quality_score,
        ),
        reverse=True,
    )

    selected: List[SearchCandidate] = []
    doc_counts: Dict[str, int] = defaultdict(int)

    for item in enriched:
        candidate = item["candidate"]
        doc_key = normalize_text(candidate.doc_title or candidate.source_file)

        if doc_key and doc_counts[doc_key] >= max_per_document:
            continue

        duplicate = False
        for prev in selected:
            if candidate.chunk_uid == prev.chunk_uid:
                duplicate = True
                break
            if normalize_text(candidate.text_preview) == normalize_text(prev.text_preview):
                duplicate = True
                break
        if duplicate:
            continue
        if candidate.chunk_level == "child" and item["body_support_score"] < 0.08 and item["lexical_bonus"] >= 0.18:
            continue
        if item["offtopic_penalty"] >= 0.65 and item["task_alignment_score"] < 0.20:
            continue
        if plan and plan.task_type in {"fact", "definition"}:
            if item["body_support_score"] < 0.06 and item["task_alignment_score"] < 0.55:
                continue
            if item["body_support_score"] < 0.10 and item["lexical_bonus"] <= 0.02 and item["query_match_bonus"] <= 0.03:
                continue

        candidate.score_breakdown.update(
            {
                "hybrid_base": item["hybrid_base"],
                "qwen_dense": item["qwen_dense"],
                "bge_dense": item["bge_dense"],
                "lexical_bonus": item["lexical_bonus"],
                "lexical_sparse": item["lexical_sparse"],
                "lexical_sparse_raw": item["lexical_sparse_raw"],
                "bge_sparse": item["bge_sparse"],
                "bge_sparse_raw": item["bge_sparse_raw"],
                "metadata_bonus": item["metadata_bonus"],
                "query_match_bonus": item["query_match_bonus"],
                "local_rerank_score": item["local_rerank_score"],
                "rerank_score": item["rerank_score"],
                "parent_recall_score": item["parent_recall_score"],
                "body_support_score": item["body_support_score"],
                "plan_alignment": item["plan_alignment"],
                "task_alignment_score": item["task_alignment_score"],
                "facet_coverage_score": item["facet_coverage_score"],
                "doctype_fitness_score": item["doctype_fitness_score"],
                "offtopic_penalty": item["offtopic_penalty"],
                "final_score": item["final_score"],
            }
        )

        selected.append(candidate)
        if doc_key:
            doc_counts[doc_key] += 1
        if len(selected) >= max(1, top_k):
            break

    selected.sort(key=lambda c: c.score_breakdown.get("final_score", 0.0), reverse=True)
    return selected


def evaluate_evidence_coverage(plan: QueryPlan, evidence_items: Sequence[EvidenceItem]) -> Dict[str, Any]:
    required_facets = set(get_task_required_facets(plan.task_type))
    if not required_facets:
        return {
            "passed": bool(evidence_items),
            "covered": [],
            "missing": [],
            "min_required": 1,
        }

    covered: set[str] = set()
    for item in evidence_items:
        text = normalize_text(" ".join([item.doc_title, item.section_title, item.quote]))
        covered.update(get_task_facets_from_text(text, plan.task_type) & required_facets)

    min_required = min(_TASK_MIN_COVERAGE.get(plan.task_type, 1), len(required_facets))
    return {
        "passed": len(covered) >= min_required,
        "covered": sorted(covered),
        "missing": sorted(required_facets - covered),
        "min_required": min_required,
    }


def evaluate_topic_consistency(plan: QueryPlan, evidence_items: Sequence[EvidenceItem]) -> Dict[str, Any]:
    anchors = build_topic_anchor_terms(plan)
    if not anchors:
        return {"passed": bool(evidence_items), "anchors": [], "aligned": len(evidence_items), "drifted": []}
    aligned = 0
    drifted: List[Dict[str, str]] = []
    for item in evidence_items:
        text = normalize_text(" ".join([item.doc_title, item.section_title, item.quote]))
        if any(anchor in text for anchor in anchors):
            aligned += 1
            continue
        drifted.append(
            {
                "citation": format_citation_label(item.citation),
                "quote": item.quote[:120],
            }
        )
    passed = aligned >= max(1, math.ceil(len(evidence_items) * 0.5))
    return {
        "passed": passed,
        "anchors": anchors,
        "aligned": aligned,
        "drifted": drifted[:4],
    }


def select_evidence(
    query: str,
    plan: QueryPlan,
    candidates: Sequence[SearchCandidate],
    evidence_top_k: int,
    max_per_document: int,
    core_evidence_top_k: int = DEFAULT_CORE_EVIDENCE_TOP_K,
) -> List[EvidenceItem]:
    ranked_items: List[Dict[str, Any]] = []
    required_facets = set(get_task_required_facets(plan.task_type))
    normalized_query = normalize_text(query)
    for candidate in candidates:
        final_score = float(candidate.score_breakdown.get("final_score", candidate.semantic_score))
        body_support_score = float(candidate.score_breakdown.get("body_support_score", compute_body_support_score(query, candidate.payload)))
        answerability_score = float(candidate.payload.get("answerability_score", 0.0) or 0.0)
        quality_score = float(candidate.payload.get("quality_score", candidate.quality_score) or 0.0)
        info_density = float(candidate.payload.get("info_density", 0.0) or 0.0)
        noise_score = float(candidate.payload.get("noise_score", 0.0) or 0.0)
        plan_alignment = score_plan_alignment(plan, candidate)
        quote = extract_best_quote(query, plan, candidate)
        if not quote:
            continue
        task_alignment = compute_task_alignment_breakdown(plan, candidate, quote=quote)
        quote_signal_score = quote_task_signal_score(plan, quote)
        quote_noise_penalty = quote_offtopic_penalty(plan, quote)
        source_noise_penalty = industry_source_noise_penalty(plan, candidate)
        source_focus_score = industry_source_focus_score(plan, candidate)
        analysis_text = build_candidate_analysis_text(candidate, quote=quote)
        query_exact_hit = bool(normalized_query and normalized_query in analysis_text)
        if task_alignment["offtopic_penalty"] >= 0.40 and task_alignment["task_alignment_score"] < 0.20:
            continue
        if quote_noise_penalty >= 0.30 and quote_signal_score < 0.45:
            continue
        if is_industry_thesis_query(plan) and plan.task_type == "trend":
            min_signal_score = 0.20 if source_focus_score >= 0.50 else 0.35
            if quote_signal_score < min_signal_score:
                continue
            if source_noise_penalty >= 0.12 and quote_signal_score < 0.70:
                continue
        if plan.task_type == "fact" and not query_exact_hit and body_support_score < 0.18 and task_alignment["anchor_ratio"] < 0.30:
            continue
        detected_facets = set(task_alignment["detected_facets"])
        coverage_bonus = 0.10 * len(detected_facets & required_facets)
        evidence_score = (
            (0.42 * final_score)
            + (0.20 * body_support_score)
            + (0.14 * answerability_score)
            + (0.10 * quality_score)
            + (0.08 * info_density)
            + (0.10 * plan_alignment)
            + (0.12 * task_alignment["task_alignment_score"])
            + (0.18 * quote_signal_score)
            + (0.12 * source_focus_score)
            + coverage_bonus
            - (0.08 * noise_score)
            - (0.10 * task_alignment["offtopic_penalty"])
            - (0.18 * quote_noise_penalty)
            - (0.14 * source_noise_penalty)
        )
        ranked_items.append(
            {
                "evidence_score": float(evidence_score),
                "candidate": candidate,
                "quote": quote,
                "body_support_score": float(body_support_score),
                "answerability_score": float(answerability_score),
                "detected_facets": detected_facets,
                "quote_signal_score": float(quote_signal_score),
                "quote_noise_penalty": float(quote_noise_penalty),
                "source_noise_penalty": float(source_noise_penalty),
                "source_focus_score": float(source_focus_score),
            }
        )

    ranked_items.sort(
        key=lambda item: (
            item["evidence_score"],
            len(item["detected_facets"]),
            item["candidate"].score_breakdown.get("final_score", 0.0),
        ),
        reverse=True,
    )

    selected: List[EvidenceItem] = []
    selected_chunk_ids = set()
    doc_counts: Dict[str, int] = defaultdict(int)
    parent_counts: Dict[str, int] = defaultdict(int)
    seen_quotes = set()
    covered_required: set[str] = set()
    min_coverage = min(_TASK_MIN_COVERAGE.get(plan.task_type, 1), len(required_facets)) if required_facets else 0
    selected_focus_count = 0
    focus_source_minimum = min(max(3, min_coverage + 1), max(1, evidence_top_k))

    for prefer_coverage in (True, False):
        for item in ranked_items:
            candidate = item["candidate"]
            quote = item["quote"]
            evidence_score = item["evidence_score"]
            body_support_score = item["body_support_score"]
            answerability_score = item["answerability_score"]
            detected_facets = item["detected_facets"]
            source_focus_score = float(item.get("source_focus_score", 0.0) or 0.0)

            if candidate.chunk_uid in selected_chunk_ids:
                continue
            doc_key = normalize_text(candidate.doc_title or candidate.source_file)
            parent_key = candidate.parent_chunk_uid or candidate.chunk_uid
            quote_key = normalize_text(quote)
            adds_new_required = bool((detected_facets & required_facets) - covered_required)
            doc_limit = max(2, min(max_per_document, 3))
            parent_limit = 2
            if is_industry_thesis_query(plan) and plan.task_type == "trend":
                if industry_source_focus_score(plan, candidate) >= 0.50:
                    doc_limit = max(doc_limit, min(max_per_document, 6))
                    parent_limit = max(parent_limit, min(max_per_document, 6))

            if prefer_coverage and min_coverage and len(covered_required) < min_coverage and not adds_new_required:
                continue
            if (
                is_industry_thesis_query(plan)
                and plan.task_type == "trend"
                and selected_focus_count >= focus_source_minimum
                and source_focus_score < 0.50
            ):
                continue
            if doc_key and doc_counts[doc_key] >= doc_limit:
                continue
            if parent_key and parent_counts[parent_key] >= parent_limit:
                continue
            if quote_key in seen_quotes:
                continue

            seen_quotes.add(quote_key)
            selected_chunk_ids.add(candidate.chunk_uid)
            covered_required.update(detected_facets & required_facets)
            group = assign_evidence_group(plan, candidate, quote)
            selected.append(
                EvidenceItem(
                    rank=len(selected) + 1,
                    chunk_uid=candidate.chunk_uid,
                    source_file=candidate.source_file,
                    doc_title=candidate.doc_title,
                    section_title=candidate.section_title,
                    chunk_level=candidate.chunk_level,
                    group=group,
                    quote=quote,
                    evidence_score=float(evidence_score),
                    body_support_score=float(body_support_score),
                    answerability_score=float(answerability_score),
                    final_score=float(candidate.score_breakdown.get("final_score", candidate.semantic_score)),
                    citation={
                        "chunk_uid": candidate.chunk_uid,
                        "source_file": candidate.source_file,
                        "doc_title": candidate.doc_title,
                        "section_title": candidate.section_title,
                    },
                )
            )
            if doc_key:
                doc_counts[doc_key] += 1
            if parent_key:
                parent_counts[parent_key] += 1
            if source_focus_score >= 0.50:
                selected_focus_count += 1
            if len(selected) >= max(1, evidence_top_k):
                break
        if len(selected) >= max(1, evidence_top_k):
            break

    target_evidence_count = max(1, evidence_top_k)
    if len(selected) < target_evidence_count:
        for candidate in candidates:
            if len(selected) >= target_evidence_count:
                break
            if candidate.chunk_uid in selected_chunk_ids:
                continue
            doc_key = normalize_text(candidate.doc_title or candidate.source_file)
            parent_key = candidate.parent_chunk_uid or candidate.chunk_uid
            if doc_key and doc_counts[doc_key] >= max(1, max_per_document):
                continue
            quote = extract_best_quote(query, plan, candidate) or cleanup_retrieval_text(candidate.text_preview)
            quote = quote[:600].strip()
            if not quote:
                continue
            quote_key = normalize_text(quote)
            if quote_key in seen_quotes:
                continue

            final_score = float(candidate.score_breakdown.get("final_score", candidate.semantic_score))
            body_support_score = float(
                candidate.score_breakdown.get("body_support_score", compute_body_support_score(query, candidate.payload))
            )
            answerability_score = float(candidate.payload.get("answerability_score", 0.0) or 0.0)
            seen_quotes.add(quote_key)
            selected_chunk_ids.add(candidate.chunk_uid)
            selected.append(
                EvidenceItem(
                    rank=len(selected) + 1,
                    chunk_uid=candidate.chunk_uid,
                    source_file=candidate.source_file,
                    doc_title=candidate.doc_title,
                    section_title=candidate.section_title,
                    chunk_level=candidate.chunk_level,
                    group=assign_evidence_group(plan, candidate, quote),
                    quote=quote,
                    evidence_score=final_score,
                    body_support_score=body_support_score,
                    answerability_score=answerability_score,
                    final_score=final_score,
                    citation={
                        "chunk_uid": candidate.chunk_uid,
                        "source_file": candidate.source_file,
                        "doc_title": candidate.doc_title,
                        "section_title": candidate.section_title,
                    },
                )
            )
            if doc_key:
                doc_counts[doc_key] += 1
            if parent_key:
                parent_counts[parent_key] += 1

    for index, item in enumerate(selected, start=1):
        item.rank = index
        item.tier = "core" if index <= max(1, int(core_evidence_top_k)) else "support"
    return selected


def print_results(
    query: str,
    standalone_query: str,
    intent: str,
    query_variants: Sequence[str],
    results: Sequence[SearchCandidate],
    plan: Optional[QueryPlan] = None,
    evidence_items: Optional[Sequence[EvidenceItem]] = None,
    answer: Optional[AnswerSynthesis] = None,
    timings: Optional[Dict[str, float]] = None,
    trace_file: str = "",
    notices: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> None:
    if standalone_query and standalone_query != query:
        print(f"Standalone query: {standalone_query}")
    print(f"查询: {query}")
    print(f"意图: {intent}")
    print(f"查询变体: {list(query_variants)}")
    if plan:
        print(
            "检索规划: "
            f"task_type={plan.task_type} "
            f"multi_hop={plan.needs_multi_hop} "
            f"theme_terms={plan.theme_terms} "
            f"entity_terms={plan.entity_terms} "
            f"time_terms={plan.time_terms} "
            f"focus={plan.evidence_focus} "
            f"filter_hints={plan.filter_hints}"
        )
        if plan.notes:
            print(f"规划备注: {plan.notes}")
    print(f"命中数: {len(results)}")
    if answer:
        print("[答案]")
        print(f"状态: {answer.status}  置信度: {answer.confidence:.4f}")
        if answer.refusal_reason:
            print(f"拒答原因: {answer.refusal_reason}")
        if answer.review_status:
            print(f"审查状态: {answer.review_status}")
        if answer.review_issues:
            print(f"审查问题: {json.dumps(answer.review_issues, ensure_ascii=False)}")
        if answer.conflicts:
            print(f"冲突提示: {json.dumps(answer.conflicts, ensure_ascii=False)}")
        print(answer.answer)
    elif not evidence_items:
        print("[答案]")
        print("当前没有足够证据生成回答。")
    if evidence_items:
        print("[答案证据]")
        for item in evidence_items:
            label = format_citation_label(item.citation)
            print(f"{item.rank}. {label}")
            print(f"   {item.quote}")
    if verbose:
        if timings:
            print(f"[耗时] {json.dumps(timings, ensure_ascii=False)}")
        print("=" * 120)
        for idx, c in enumerate(results, start=1):
            header_path = " > ".join(c.header_path) if c.header_path else "(none)"
            breakdown = c.score_breakdown
            print(f"[{idx}] final_score={breakdown.get('final_score', 0.0):.4f} chunk_uid={c.chunk_uid}")
            print(
                f"source={c.source_name} "
                f"qwen_dense={breakdown.get('qwen_dense', 0.0):.4f} "
                f"qwen_dense_flat_child={breakdown.get('qwen_dense_flat_child', 0.0):.4f} "
                f"bge_dense={breakdown.get('bge_dense', 0.0):.4f} "
                f"bge_dense_flat_child={breakdown.get('bge_dense_flat_child', 0.0):.4f} "
                f"lexical_sparse={breakdown.get('lexical_sparse', 0.0):.4f} "
                f"lexical_sparse_flat_child={breakdown.get('lexical_sparse_flat_child', 0.0):.4f} "
                f"bge_sparse={breakdown.get('bge_sparse', 0.0):.4f} "
                f"bge_sparse_flat_child={breakdown.get('bge_sparse_flat_child', 0.0):.4f} "
                f"fusion_rrf={breakdown.get('fusion_rrf', 0.0):.4f} "
                f"api_rerank_score={breakdown.get('api_rerank_score', 0.0):.4f} "
                f"local_rerank_score={breakdown.get('local_rerank_score', 0.0):.4f} "
                f"lexical_bonus={breakdown.get('lexical_bonus', 0.0):.4f} "
                f"metadata_bonus={breakdown.get('metadata_bonus', 0.0):.4f} "
                f"body_support_score={breakdown.get('body_support_score', 0.0):.4f} "
                f"task_alignment_score={breakdown.get('task_alignment_score', 0.0):.4f} "
                f"offtopic_penalty={breakdown.get('offtopic_penalty', 0.0):.4f}"
            )
            print(f"doc_title: {c.doc_title}")
            print(f"source_file: {c.source_file}")
            print(f"section_title: {c.section_title}")
            print(f"chunk_level: {c.chunk_level}")
            print(f"chunk_type: {c.chunk_type}")
            print(f"quality_score: {c.quality_score:.2f}")
            print(
                f"page={c.payload.get('page_label') or c.payload.get('page_no') or ''} "
                f"unit={c.payload.get('knowledge_unit_type', '')} "
                f"info_density={float(c.payload.get('info_density', 0.0) or 0.0):.2f} "
                f"noise_score={float(c.payload.get('noise_score', 0.0) or 0.0):.2f}"
            )
            print(f"header_path: {header_path}")
            print(f"matched_queries: {c.matched_queries}")
            print(f"text_preview: {c.text_preview}")
            print("-" * 120)


def normalize_answer_text(text: str) -> str:
    return re.sub(r"[，。！？；;:：“”\"'（）()\[\]【】《》、\s]+", "", normalize_text(text))


def answer_text_shingles(text: str, width: int = 2) -> set[str]:
    normalized = normalize_answer_text(text)
    if len(normalized) <= width:
        return {normalized} if normalized else set()
    return {normalized[index : index + width] for index in range(len(normalized) - width + 1)}


def answer_text_is_similar(left: str, right: str) -> bool:
    left_normalized = normalize_answer_text(left)
    right_normalized = normalize_answer_text(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized in right_normalized or right_normalized in left_normalized:
        return True
    left_shingles = answer_text_shingles(left_normalized)
    right_shingles = answer_text_shingles(right_normalized)
    if not left_shingles or not right_shingles:
        return False
    overlap = len(left_shingles & right_shingles) / max(len(left_shingles | right_shingles), 1)
    return overlap >= 0.55


def answer_text_overlap_ratio(left: str, right: str) -> float:
    left_shingles = answer_text_shingles(left)
    right_shingles = answer_text_shingles(right)
    if not left_shingles or not right_shingles:
        return 0.0
    return len(left_shingles & right_shingles) / max(min(len(left_shingles), len(right_shingles)), 1)


def split_quote_into_answer_nuggets(quote: str) -> List[str]:
    cleaned = cleanup_retrieval_text(quote)
    if not cleaned:
        return []
    parts: List[str] = []
    for sentence in split_sentences(cleaned):
        for piece in re.split(r"[；;]", sentence):
            compact = cleanup_retrieval_text(piece).strip("，。；;:： ")
            if not compact:
                continue
            if count_cjk(compact) < 4 and len(compact) < 16:
                continue
            parts.append(compact)
    return unique_preserve(parts, max_items=6)


def cleanup_answer_nugget(text: str) -> str:
    cleaned = cleanup_retrieval_text(text)
    cleaned = cleaned.replace("K动", "推动")
    cleaned = re.sub(r"^(?:此外|另外|同时|其次|再次|最后|第一|第二|第三|其一|其二|其三|具体而言|其中)\s*[，,:：]?", "", cleaned)
    cleaned = re.sub(r"^(?:基金优势在于|其优势在于)", "", cleaned)
    cleaned = re.sub(r"^这主要体现在", "主要体现在", cleaned)
    cleaned = re.sub(r"^主要体现在通过", "主要通过", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip("，。；;:： ")
    return cleaned


def is_aspect_like_query(plan: QueryPlan) -> bool:
    lowered = normalize_text(plan.original_query)
    return contains_any(lowered, ["优势", "体现在", "方面", "哪些", "哪几个", "包括什么", "主要包括"])


def score_answer_nugget(plan: QueryPlan, evidence_item: EvidenceItem, nugget: str) -> float:
    lowered = normalize_text(nugget)
    anchors = build_topic_anchor_terms(plan)
    facets = get_task_facets_from_text(nugget, plan.task_type)
    score = float(evidence_item.evidence_score)
    score += 0.08 * sum(1 for anchor in anchors if anchor in lowered)
    score += 0.07 * len(facets)
    if contains_any(lowered, ["包括", "主要", "在于", "体现在", "分为", "涵盖", "通过", "重点"]):
        score += 0.06
    if contains_any(lowered, ["例如", "比如", "案例", "涉及的企业包括", "项目包括"]):
        score -= 0.08
    if len(re.findall(r"\d", nugget)) > 12 and plan.task_type in {"fact", "definition", "status"}:
        score -= 0.12
    if len(nugget) > 120:
        score -= 0.04
    if is_aspect_like_query(plan):
        if contains_any(lowered, ["拥有", "提供", "突出", "可靠", "来源", "服务", "地位", "团队", "能力", "渠道"]):
            score += 0.10
        if 8 <= count_cjk(nugget) <= 30:
            score += 0.05
        if contains_any(lowered, ["具体", "案例", "例如", "ipo项目", "拟上市项目"]):
            score -= 0.05
    if plan.task_type == "trend" and is_industry_thesis_query(plan):
        score += 0.16 * quote_task_signal_score(plan, nugget)
        score -= 0.14 * quote_offtopic_penalty(plan, nugget)
    return score


def should_skip_answer_nugget(plan: QueryPlan, nugget: str) -> bool:
    lowered = normalize_text(nugget)
    if not lowered:
        return True
    if count_cjk(nugget) < 4 and len(nugget) < 16:
        return True
    if contains_any(lowered, ["document:", "content type:", "pptx", "逻辑文本"]):
        return True
    if plan.task_type in {"fact", "definition"} and contains_any(lowered, ["例如", "案例", "项目包括", "涉及的企业包括"]):
        return True
    if plan.task_type == "fact" and len(re.findall(r"\d", nugget)) > 16:
        return True
    if plan.task_type == "trend" and is_industry_thesis_query(plan):
        if quote_offtopic_penalty(plan, nugget) >= 0.35 and quote_task_signal_score(plan, nugget) < 0.30:
            return True
    return False


def claim_intro_text(plan: QueryPlan, conflicts: Sequence[Dict[str, Any]]) -> str:
    if conflicts:
        return "检索到了可用证据，但不同来源里的时间或数值细节不完全一致，以下结论按保守口径整理："
    if plan.task_type == "procedure":
        return "结合当前证据，可以按以下步骤或要点处理："
    if plan.task_type == "comparison":
        return "结合当前证据，核心差异可以归纳为："
    if plan.task_type == "root_cause":
        return "结合当前证据，主要原因可以归纳为："
    if plan.task_type == "trend":
        return "结合当前证据，主要趋势可以归纳为："
    if plan.task_type == "market":
        return "结合当前证据，当前可确认的关键信息包括："
    return "根据当前检索到的证据，可以归纳出以下要点："


def merge_claim_citations(citations: Sequence[Dict[str, str]], max_citations: int = 2) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    seen = set()
    for citation in citations:
        label = format_citation_label(citation)
        if not label or label in seen:
            continue
        seen.add(label)
        merged.append(citation)
        if len(merged) >= max(1, max_citations):
            break
    return merged


def answer_claim_priority(plan: QueryPlan, claim: Dict[str, Any]) -> float:
    lowered = normalize_text(claim.get("text", ""))
    priority = float(claim.get("score", claim.get("confidence", 0.0)))
    if is_aspect_like_query(plan):
        if contains_any(lowered, ["拥有", "提供", "突出", "可靠", "来源", "服务", "地位", "团队", "能力"]):
            priority += 0.12
        if contains_any(lowered, ["具体", "渠道", "案例", "例如", "ipo项目", "拟上市项目"]):
            priority -= 0.06
    if plan.task_type == "trend" and is_industry_thesis_query(plan):
        priority += 0.18 * quote_task_signal_score(plan, lowered)
        priority -= 0.16 * quote_offtopic_penalty(plan, lowered)
    return priority


def build_answer_claims(plan: QueryPlan, evidence_items: Sequence[EvidenceItem], max_claims: int) -> List[Dict[str, Any]]:
    candidate_claims: List[Dict[str, Any]] = []

    for item in evidence_items:
        for raw_nugget in split_quote_into_answer_nuggets(item.quote):
            nugget = cleanup_answer_nugget(raw_nugget)
            if should_skip_answer_nugget(plan, nugget):
                continue
            score = score_answer_nugget(plan, item, nugget)
            candidate_claims.append(
                {
                    "text": nugget,
                    "citations": [item.citation],
                    "group": item.group,
                    "confidence": round(max(item.evidence_score, score), 4),
                    "score": score,
                }
            )

    merged_claims: List[Dict[str, Any]] = []
    for claim in sorted(candidate_claims, key=lambda value: (value["score"], len(value["text"])), reverse=True):
        merged = False
        for existing in merged_claims:
            if answer_text_is_similar(claim["text"], existing["text"]):
                if len(claim["text"]) > len(existing["text"]) and claim["score"] >= existing["score"] - 0.05:
                    existing["text"] = claim["text"]
                existing["citations"] = merge_claim_citations(existing["citations"] + claim["citations"], max_citations=3)
                existing["confidence"] = round(max(existing["confidence"], claim["confidence"]), 4)
                existing["score"] = max(existing["score"], claim["score"])
                merged = True
                break
        if not merged:
            merged_claims.append(claim)

    merged_claims.sort(
        key=lambda value: (answer_claim_priority(plan, value), value["confidence"], len(value["text"])),
        reverse=True,
    )
    final_claims: List[Dict[str, Any]] = []
    claim_limit = max(1, max_claims)
    if plan.task_type == "trend" and is_industry_thesis_query(plan):
        claim_limit = max(claim_limit, max_claims + 3)
    for claim in merged_claims:
        if any(answer_text_is_similar(claim["text"], existing["text"]) for existing in final_claims):
            continue
        if is_aspect_like_query(plan) and any(answer_text_overlap_ratio(claim["text"], existing["text"]) >= 0.30 for existing in final_claims):
            continue
        final_claims.append(
            {
                "text": claim["text"],
                "citations": merge_claim_citations(claim["citations"], max_citations=2),
                "group": claim["group"],
                "confidence": round(claim["confidence"], 4),
            }
        )
        if len(final_claims) >= claim_limit:
            break
    return final_claims


def infer_claim_label(plan: QueryPlan, text: str, index: int) -> str:
    lowered = normalize_text(text)
    if is_aspect_like_query(plan):
        if contains_any(lowered, ["来源", "渠道", "中介网络", "项目获取", "项目来源"]):
            return "项目来源"
        if contains_any(lowered, ["投后", "赋能", "服务", "智慧赋能"]):
            return "投后赋能"
        if contains_any(lowered, ["资源", "平台", "战略投资方", "集团"]):
            return "资源平台"
        if contains_any(lowered, ["行业地位", "竞争格局", "行业研究"]):
            return "行业地位"
        if contains_any(lowered, ["团队", "企业家", "创新能力", "组织能力"]):
            return "团队能力"
        return f"优势{index}"

    if plan.task_type == "market":
        if contains_any(lowered, ["规模", "销量", "销售额", "出货量"]):
            return "规模"
        if contains_any(lowered, ["增速", "增长", "趋势", "变化"]):
            return "趋势"
        if contains_any(lowered, ["份额", "格局", "厂商", "集中度"]):
            return "格局"
        return f"市场要点{index}"
    if plan.task_type == "comparison":
        return f"差异点{index}"
    if plan.task_type == "procedure":
        return f"步骤{index}"
    if plan.task_type == "root_cause":
        return f"原因{index}"
    if plan.task_type == "trend":
        if is_industry_thesis_query(plan):
            if contains_any(lowered, ["技术创新", "核心驱动力", "智能技术", "驱动力"]):
                return "技术驱动"
            if contains_any(lowered, ["制造业", "互联网", "融合", "智能制造", "推动产业"]):
                return "产业融合"
            if contains_any(lowered, ["机器学习", "视觉识别", "语音识别", "自动推理", "人工智能"]):
                return "应用能力"
            if contains_any(lowered, ["未来", "趋势", "变革", "革命"]):
                return "趋势判断"
        return f"趋势{index}"
    if plan.task_type == "definition":
        return f"定义要点{index}"
    if contains_any(lowered, ["主要包含", "包含", "组成", "构成"]):
        return "结构构成"
    if contains_any(lowered, ["指导方针", "产融结合", "政府基金", "社会资本"]):
        return "投资方针"
    if contains_any(lowered, ["市场化运作", "专业化管理", "投资收益"]):
        return "运作方式"
    return f"要点{index}"


def build_answer_summary(plan: QueryPlan, claims: Sequence[Dict[str, Any]]) -> str:
    labels = unique_preserve(
        [infer_claim_label(plan, claim.get("text", ""), index + 1) for index, claim in enumerate(claims)],
        max_items=4,
    )
    if is_aspect_like_query(plan):
        if labels:
            return f"简要判断：当前证据显示，优势主要集中在{'、'.join(labels)}。"
        return "简要判断：当前证据显示，优势可以拆成若干具体方面。"
    if plan.task_type == "market":
        return "核心判断：当前证据可先按行研框架，从市场空间、增长趋势、竞争格局和盈利/风险变量理解，缺口维度应继续补检索。"
    if plan.task_type == "comparison":
        return "简要判断：当前证据支持按差异点逐项比较，不宜只给单一结论。"
    if plan.task_type == "procedure":
        return "简要判断：当前证据更适合整理成执行步骤和注意事项。"
    if plan.task_type == "root_cause":
        return "简要判断：当前证据支持把原因拆成若干驱动因素，而不是归因到单一因素。"
    if plan.task_type == "trend":
        if is_industry_thesis_query(plan):
            return "核心判断：当前证据倾向于支持人工智能/智能技术具备引领下一轮产业变革的条件，但结论需要结合技术驱动、产业融合和应用落地一起看。"
        return "核心判断：当前证据更适合按趋势表现、驱动因素、商业化节奏和时间口径来读。"
    if plan.task_type == "definition":
        return "简要判断：当前证据可以先给出定义，再补充核心特征。"
    return "核心判断：当前证据可以支持以下事实性要点。"


def merge_claims_by_label_for_render(plan: QueryPlan, claims: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for idx, claim in enumerate(claims, start=1):
        label = infer_claim_label(plan, claim.get("text", ""), idx)
        text = str(claim.get("text", "")).strip("。；; ")
        if not text:
            continue
        bucket = grouped.get(label)
        if bucket is None:
            bucket = {
                "label": label,
                "segments": [],
                "citations": [],
                "confidence": 0.0,
                "group": claim.get("group", ""),
            }
            grouped[label] = bucket
            order.append(label)
        if not any(answer_text_is_similar(text, existing) for existing in bucket["segments"]):
            bucket["segments"].append(text)
        bucket["citations"] = merge_claim_citations(bucket["citations"] + claim.get("citations", []), max_citations=3)
        bucket["confidence"] = max(float(bucket.get("confidence", 0.0)), float(claim.get("confidence", 0.0) or 0.0))

    merged: List[Dict[str, Any]] = []
    for label in order:
        bucket = grouped[label]
        segments = bucket["segments"][:2]
        if not segments:
            continue
        merged.append(
            {
                "label": label,
                "text": "；".join(segments),
                "citations": merge_claim_citations(bucket["citations"], max_citations=2),
                "group": bucket.get("group", ""),
                "confidence": round(float(bucket.get("confidence", 0.0)), 4),
            }
        )
    return merged


def render_grounded_answer_lines(
    plan: QueryPlan,
    claims: Sequence[Dict[str, Any]],
    conflicts: Sequence[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    if conflicts:
        lines.append("注意：不同来源里的时间或数值细节不完全一致，以下按保守口径整理。")
    lines.append(build_answer_summary(plan, claims))

    if is_aspect_like_query(plan):
        lines.append("分项依据：")
    elif plan.task_type in {"market", "comparison", "procedure", "root_cause", "trend"}:
        lines.append("关键依据：")
    else:
        lines.append("支撑要点：")

    should_merge_claims = is_aspect_like_query(plan) or (plan.task_type == "trend" and is_industry_thesis_query(plan))
    render_claims = merge_claims_by_label_for_render(plan, claims) if should_merge_claims else list(claims)

    for idx, claim in enumerate(render_claims, start=1):
        label = claim.get("label") or infer_claim_label(plan, claim.get("text", ""), idx)
        citation_labels = [format_citation_label(citation) for citation in claim.get("citations", [])]
        citation_labels = [item for item in unique_preserve(citation_labels, max_items=2) if item]
        citation_text = "；".join(citation_labels)
        text = str(claim.get("text", "")).strip("。；; ")
        if citation_text:
            lines.append(f"{idx}. {label}：{text}（证据：{citation_text}）")
        else:
            lines.append(f"{idx}. {label}：{text}")
    return lines


def synthesize_grounded_answer(
    query: str,
    plan: QueryPlan,
    evidence_items: Sequence[EvidenceItem],
    min_evidence: int,
    min_evidence_score: float,
    max_claims: int,
) -> AnswerSynthesis:
    supported = [item for item in evidence_items if item.evidence_score >= min_evidence_score]
    avg_score = sum(item.evidence_score for item in evidence_items) / max(len(evidence_items), 1)
    if len(supported) < max(1, min_evidence):
        return AnswerSynthesis(
            status="insufficient_evidence",
            confidence=round(min(0.49, avg_score), 4),
            answer="当前证据还不够强，暂时不能可靠回答。建议补充更明确的主体、时间范围或检索条件后再查一次。",
            refusal_reason="not_enough_high_confidence_evidence",
        )

    coverage = evaluate_evidence_coverage(plan, supported)
    if not coverage["passed"]:
        covered_text = "、".join(coverage["covered"]) or "暂无"
        missing_text = "、".join(coverage["missing"]) or "暂无"
        return AnswerSynthesis(
            status="insufficient_evidence",
            confidence=round(min(0.54, avg_score), 4),
            answer=(
                f"当前证据还不足以覆盖这个问题需要的关键维度。"
                f"已覆盖：{covered_text}；仍缺少：{missing_text}。"
                "建议补充更明确的主体、时间范围或过滤条件后再检索。"
            ),
            refusal_reason="insufficient_task_coverage",
        )

    topic_consistency = evaluate_topic_consistency(plan, supported)
    aligned_supported = supported
    if topic_consistency["anchors"]:
        aligned_supported = [
            item
            for item in supported
            if any(
                anchor in normalize_text(" ".join([item.doc_title, item.section_title, item.quote]))
                for anchor in topic_consistency["anchors"]
            )
        ]
    if not topic_consistency["passed"] and len(aligned_supported) < max(1, min_evidence):
        return AnswerSynthesis(
            status="cautious_answer",
            confidence=round(min(0.58, avg_score), 4),
            answer="检索到了一些相关材料，但证据主题还不够一致，系统先不直接生成结论。建议补充更明确的主体或约束条件后再查。",
            conflicts=[
                {
                    "type": "topic_drift",
                    "message": "Selected evidence does not stay on the same subject.",
                    "anchors": topic_consistency["anchors"],
                    "drifted": topic_consistency["drifted"],
                }
            ],
            refusal_reason="topic_drift_detected",
        )

    conflicts = detect_evidence_conflicts(query, aligned_supported)
    claims = build_answer_claims(plan, aligned_supported, max_claims=max_claims)
    if not claims:
        return AnswerSynthesis(
            status="insufficient_evidence",
            confidence=round(min(0.49, avg_score), 4),
            answer="当前检索结果里还没有足够清晰、可直接支撑结论的句子，所以系统先不直接回答。",
            refusal_reason="no_groundable_claims",
        )

    answer_lines = render_grounded_answer_lines(plan, claims, conflicts)

    confidence = min(
        0.95,
        (
            sum(item.evidence_score for item in aligned_supported[:max_claims]) / max(len(aligned_supported[:max_claims]), 1)
        ) + (0.03 * len(coverage["covered"])),
    )
    return AnswerSynthesis(
        status="conflicted" if conflicts else "answered",
        confidence=round(confidence, 4),
        answer="\n".join(answer_lines),
        claims=claims,
        conflicts=conflicts,
    )


_DOMAIN_QUERY_TERMS = [
    "人工智能",
    "产业革命",
    "产业变革",
    "工业革命",
    "智能技术",
    "智能制造",
    "技术创新",
    "制造业",
    "互联网",
    "机器人",
    "工业机器人",
    "云计算",
    "大数据",
    "机器学习",
    "智能产业",
]


def extract_domain_terms(query: str) -> List[str]:
    lowered = normalize_text(query)
    terms = [term for term in _DOMAIN_QUERY_TERMS if term in lowered]
    if "ai" in lowered or "aigc" in lowered:
        terms.append("人工智能")
    if "下一个产业革命" in lowered or "下个产业革命" in lowered or "下一轮产业革命" in lowered:
        terms.append("产业革命")
    if "产业革命浪潮" in lowered or "产业浪潮" in lowered:
        terms.append("产业变革")
    return unique_preserve(terms, max_items=8)


def build_query_terms(query: str) -> List[str]:
    normalized = normalize_text(query)
    if not normalized:
        return []
    domain_terms = extract_domain_terms(normalized)
    if " " in normalized:
        return unique_preserve(domain_terms + [part for part in normalized.split(" ") if len(part) >= 2])
    if _CJK_RE.search(normalized) and len(normalized) >= 2:
        segments: List[str] = []
        for part in _PUNCT_SPLIT_RE.split(normalized):
            if not part:
                continue
            cleaned = _QUERY_BREAK_RE.sub(" ", part)
            cleaned = re.sub(
                r"(?:能否|能不能|会不会|有没有|是不是|吗|嘛|呢|么|请问|下一个|下个|下一轮|引领|带来|成为|形成|推动|驱动|浪潮)",
                " ",
                cleaned,
            )
            for seg in cleaned.split():
                seg = seg.strip()
                seg = _LEADING_REFERENCE_RE.sub("", seg).strip()
                seg = cleanup_modal_suffix(seg)
                if len(seg) >= 2:
                    segments.append(seg)
        if domain_terms or segments:
            return unique_preserve(domain_terms + sorted(segments, key=len, reverse=True), max_items=12)
        return [normalized[i : i + 2] for i in range(max(1, len(normalized) - 1))]
    return unique_preserve(domain_terms + [normalized])


def extract_subject_terms(query: str) -> List[str]:
    lowered = normalize_text(query)
    candidates: List[str] = extract_domain_terms(lowered)
    cleanup_tokens = sorted(
        _GENERIC_QUERY_STOP_TOKENS
        | {"吗", "嘛", "呢", "能否", "会不会", "下个", "下一个", "下一轮", "引领", "推动", "驱动", "浪潮"},
        key=len,
        reverse=True,
    )
    for token in cleanup_tokens:
        lowered = lowered.replace(token, " ")
    for part in _PUNCT_SPLIT_RE.split(lowered):
        part = _QUERY_BREAK_RE.sub(" ", part)
        for seg in part.split():
            seg = seg.strip()
            seg = _LEADING_REFERENCE_RE.sub("", seg).strip()
            seg = cleanup_modal_suffix(seg)
            if len(seg) >= 2:
                candidates.append(seg)
    return unique_preserve(candidates, max_items=8)


def classify_query_task(query: str, intent: str) -> str:
    lowered = normalize_text(query)
    if contains_any(lowered, ["对比", "区别", "差异", "相比", "优劣", "vs", "versus", "不同"]):
        return "comparison"
    if contains_any(lowered, ["为什么", "原因", "导致", "为何", "根因", "依据", "影响因素"]):
        return "root_cause"
    if contains_any(lowered, ["是什么", "指什么", "含义", "定义", "概念", "介绍", "说明一下"]):
        return "definition"
    if contains_any(lowered, ["引领", "产业革命", "产业变革", "浪潮", "变革", "革命", "趋势", "变化", "增速", "走向", "发展", "预测", "未来", "驱动", "智能技术", "cagr", "复合增长"]):
        return "trend"
    if contains_any(
        lowered,
        [
            "市场",
            "规模",
            "份额",
            "销量",
            "出货量",
            "竞争格局",
            "行业格局",
            "渗透率",
            "需求",
            "供给",
            "赛道",
            "行研",
            "行业研究",
            "产业链",
            "价值链",
            "商业模式",
            "投资价值",
            "成长空间",
            "毛利率",
            "盈利能力",
            "成本曲线",
            "壁垒",
        ],
    ):
        return "market"
    if contains_any(lowered, ["现状", "当前", "目前", "进展", "情况", "状态", "有没有", "是否", "最新"]):
        return "status"
    if contains_any(lowered, ["流程", "步骤", "怎么办", "怎么做", "如何做", "操作", "办理", "部署", "接入"]):
        return "procedure"
    if intent == "technical" and contains_any(lowered, ["报错", "错误", "异常", "失败", "配置", "参数", "接口"]):
        return "procedure"
    return "fact"


def build_task_specific_queries(query: str, task_type: str, entity_terms: Sequence[str], time_terms: Sequence[str]) -> List[str]:
    raw_subject_terms = unique_preserve(entity_terms or extract_subject_terms(query), max_items=5)
    facet_like_terms = {
        "空间",
        "市场空间",
        "规模",
        "市场规模",
        "竞争格局",
        "行业格局",
        "毛利率",
        "盈利能力",
        "增速",
        "渗透率",
        "供需",
        "风险",
    }
    subject_terms = [
        term
        for term in raw_subject_terms
        if normalize_text(term) not in facet_like_terms and not contains_any(term, ["行研", "行业研究"])
    ][:3] or raw_subject_terms[:3]
    subject = " ".join(subject_terms) or str(query or "").strip()
    suffixes: List[str] = []
    explicit_queries: List[str] = []
    if task_type == "market":
        suffixes = ["市场规模", "增速 渗透率", "竞争格局", "供需", "产业链", "商业模式", "盈利能力 毛利率", "政策 风险"]
    elif task_type == "comparison":
        suffixes = ["区别", "差异", "优劣", "适用场景", "核心差别"]
    elif task_type == "definition":
        suffixes = ["定义", "概念", "是什么", "核心特征"]
    elif task_type == "trend":
        suffixes = ["趋势", "变化", "增长", "驱动因素", "未来", "商业化落地", "产业链影响", "政策 风险", "产业变革", "产业革命"]
        if "人工智能" in subject_terms:
            explicit_queries.extend(
                [
                    "人工智能 产业革命",
                    "人工智能 产业变革",
                    "人工智能 智能技术",
                    "人工智能 技术创新 驱动力",
                    "人工智能 制造业 互联网 智能制造",
                    "AI industrial revolution",
                ]
            )
    elif task_type == "procedure":
        suffixes = ["流程", "步骤", "要求", "注意事项", "操作方法"]
    elif task_type == "root_cause":
        suffixes = ["原因", "驱动因素", "影响因素", "导致", "依据"]
    elif task_type == "status":
        suffixes = ["现状", "当前情况", "最新进展", "主要问题"]

    queries = [query] + explicit_queries
    for suffix in suffixes:
        candidate = f"{subject} {suffix}".strip()
        if time_terms:
            candidate = f"{candidate} {' '.join(time_terms[:2])}".strip()
        queries.append(candidate)
    return unique_preserve(queries)


def build_query_plan(query: str, intent: str, max_variants: int) -> QueryPlan:
    normalized_query = normalize_text(query)
    task_type = classify_query_task(query, intent)
    base_terms = build_query_terms(query)
    subject_terms = extract_subject_terms(query)
    time_terms = extract_time_terms(query)
    theme_terms = unique_preserve(base_terms[:8])
    constraint_terms = [
        term
        for term in theme_terms
        if any(token in term for token in ["条件", "范围", "要求", "限制", "流程", "步骤", "原因", "区别", "时间"])
    ]
    entity_terms = unique_preserve(
        [
            term
            for term in (subject_terms + theme_terms)
            if term not in constraint_terms and term not in time_terms and term not in _GENERIC_QUERY_STOP_TOKENS
        ],
        max_items=8,
    )

    raw_sub_queries: List[str] = []
    for piece in re.split(r"(?:以及|同时|并且|并|和|及|对比|区别|差异|vs\.?|versus)", str(query or ""), flags=re.I):
        cleaned = " ".join(piece.split()).strip(" ,，。；;")
        if _LEADING_REFERENCE_RE.match(cleaned) and subject_terms:
            cleaned = f"{subject_terms[0]} {_LEADING_REFERENCE_RE.sub('', cleaned).strip()}".strip()
        if len(cleaned) >= 4 and normalize_text(cleaned) != normalized_query:
            raw_sub_queries.append(cleaned)

    notes: List[str] = [f"task_type={task_type}"]
    domain_terms = extract_domain_terms(query)
    if domain_terms:
        notes.append(f"domain_terms={','.join(domain_terms)}")
    needs_multi_hop = bool(_MULTI_HOP_RE.search(query)) or len(raw_sub_queries) >= 2
    if needs_multi_hop:
        notes.append("question_contains_multiple_constraints_or_subproblems")
    if time_terms:
        notes.append("question_contains_time_constraint")
    if not entity_terms:
        notes.append("entity_terms_are_weak")

    sub_queries = unique_preserve([query] + raw_sub_queries, max_items=4)
    search_queries: List[str] = []
    for candidate_query in sub_queries:
        search_queries.extend(expand_query_variants(candidate_query, max_variants=max_variants))
    search_queries.extend(build_task_specific_queries(query, task_type, entity_terms, time_terms))
    if not search_queries:
        search_queries = expand_query_variants(query, max_variants=max_variants)

    return QueryPlan(
        original_query=query,
        intent=intent,
        task_type=task_type,
        normalized_query=normalized_query,
        theme_terms=theme_terms,
        entity_terms=entity_terms,
        constraint_terms=unique_preserve(constraint_terms),
        time_terms=time_terms,
        evidence_focus=infer_evidence_focus(query, intent, task_type),
        filter_hints=infer_filter_hints(query, intent, task_type),
        sub_queries=sub_queries,
        search_queries=unique_preserve(search_queries, max_items=max(max_variants * 3, len(sub_queries) * 4, 8)),
        notes=notes,
        needs_multi_hop=needs_multi_hop,
    )


def is_industry_thesis_query(plan: QueryPlan) -> bool:
    lowered = normalize_text(plan.original_query)
    return contains_any(lowered, ["引领", "产业革命", "产业变革", "浪潮", "技术革命", "工业革命"])


def quote_task_signal_score(plan: QueryPlan, quote: str) -> float:
    lowered = normalize_text(quote)
    if not lowered:
        return 0.0
    domain_hits = sum(1 for term in extract_domain_terms(plan.original_query) if normalize_text(term) in lowered)
    anchor_hits = sum(1 for term in build_topic_anchor_terms(plan) if normalize_text(term) in lowered)
    facets = get_task_facets_from_text(quote, plan.task_type)
    score = min(0.36, 0.10 * domain_hits)
    score += min(0.24, 0.06 * anchor_hits)
    score += min(0.30, 0.10 * len(facets))

    if plan.task_type == "trend":
        ai_terms = ["人工智能", "ai", "机器学习", "视觉识别", "语音识别", "自动推理", "知识表示", "ai芯片"]
        industry_terms = ["技术创新", "核心驱动力", "智能技术", "产业变革", "产业革命", "推动产业", "主导力量", "智能制造", "制造业", "互联网"]
        ai_hits = sum(1 for term in ai_terms if term in lowered)
        industry_hits = sum(1 for term in industry_terms if term in lowered)
        score += min(0.42, 0.12 * (ai_hits + industry_hits))
        if ai_hits and industry_hits:
            score += 0.18
        if facets == {"time"}:
            score -= 0.20
    return max(0.0, min(1.0, score))


def quote_offtopic_penalty(plan: QueryPlan, quote: str) -> float:
    lowered = normalize_text(quote)
    penalty = 0.0
    if plan.task_type == "trend" and is_industry_thesis_query(plan):
        table_terms = [
            "当前市值",
            "q1",
            "现金",
            "营收",
            "估值",
            "listed",
            "market calculations",
            "capiq",
            "wall street journal",
            "排名第",
            "颜色表示",
            "星号表示",
        ]
        penalty += min(0.75, 0.15 * sum(1 for term in table_terms if term in lowered))
        investment_case_terms = ["融资", "投资", "基金", "ipo", "估值", "资本需求", "布局近", "项目源", "种子轮", "拟上市"]
        penalty += min(0.45, 0.09 * sum(1 for term in investment_case_terms if term in lowered))
    if contains_any(lowered, ["分析内容出自", "日期为", "位于第"]):
        penalty += 0.35
    return min(1.0, penalty)


def industry_source_noise_penalty(plan: QueryPlan, candidate: SearchCandidate) -> float:
    if not (plan.task_type == "trend" and is_industry_thesis_query(plan)):
        return 0.0
    source_text = normalize_text(" ".join([candidate.doc_title, candidate.source_file, candidate.section_title]))
    if not source_text:
        return 0.0
    core_title_terms = ["人工智能", "产业革命", "产业变革", "智能技术", "智能制造"]
    if any(term in source_text for term in core_title_terms):
        return 0.0
    noisy_source_terms = ["采访", "基金", "公司金融", "科创债", "阳明心学", "备案", "投资", "资本"]
    return min(0.55, 0.12 * sum(1 for term in noisy_source_terms if term in source_text))


def industry_source_focus_score(plan: QueryPlan, candidate: SearchCandidate) -> float:
    if not (plan.task_type == "trend" and is_industry_thesis_query(plan)):
        return 0.0
    source_text = normalize_text(" ".join([candidate.doc_title, candidate.source_file, candidate.section_title]))
    if not source_text:
        return 0.0
    focus_terms = ["人工智能", "产业革命", "产业变革", "智能技术", "智能制造", "ai"]
    hits = sum(1 for term in focus_terms if term in source_text)
    if hits <= 0:
        return 0.0
    return min(1.0, 0.45 + 0.12 * hits)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid enterprise search over chunk vectors stored in Qdrant.")
    parser.add_argument("query", nargs="*", default=[], help="User query used for semantic search.")
    parser.add_argument("-q", "--query-text", "--query", dest="query_text", nargs="+", default=None, help="Explicit query text. Supports multi-word questions without needing shell-specific handling.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of final hits to return.")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional minimum similarity score.")
    parser.add_argument("--db-path", default=str(DEFAULT_QDRANT_PATH), help="Local Qdrant storage directory.")
    parser.add_argument("--url", default=DEFAULT_QDRANT_URL, help="Qdrant server URL.")
    parser.add_argument("--api-key", default=DEFAULT_QDRANT_API_KEY, help="Optional Qdrant API key.")
    parser.add_argument("--prefer-grpc", action="store_true", default=DEFAULT_QDRANT_PREFER_GRPC, help="Prefer gRPC when using a Qdrant server URL.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Local Qwen embedding model path.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Embedding device, e.g. auto/cpu/cuda:0.")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, help="Embedding dtype, e.g. float16/float32/bfloat16.")
    parser.add_argument("--attn-implementation", default=DEFAULT_ATTN_IMPL, help="Transformers attention implementation.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Tokenizer max length.")
    parser.add_argument("--no-embedder-cache", action="store_true", default=not DEFAULT_KEEP_EMBEDDER_LOADED, help="Unload the local embedding model after each search call.")
    parser.add_argument("--source-file", default="", help="Optional exact-match filter for source_file.")
    parser.add_argument("--doc-title", default="", help="Optional exact-match filter for doc_title.")
    parser.add_argument("--chunk-uid", default="", help="Optional exact-match filter for chunk_uid.")
    parser.add_argument("--chunk-type", default="", help="Optional exact-match filter for chunk_type.")
    parser.add_argument("--chunk-level", default="", choices=["", "parent", "child"], help="Optional exact-match filter for chunk_level.")
    parser.add_argument("--preview-chars", type=int, default=DEFAULT_PREVIEW_CHARS, help="Characters of text preview to print.")
    parser.add_argument("--json", action="store_true", help="Print search results as JSON.")
    parser.add_argument("--answer-only", action="store_true", help="Print only the final RAG answer text.")
    parser.add_argument("--disable-instruction", action="store_true", help="Disable query-side instruction wrapping.")
    parser.add_argument("--query-variants", type=int, default=DEFAULT_QUERY_VARIANTS, help="Number of query variants to generate and fuse.")
    parser.add_argument("--candidate-multiplier", type=int, default=DEFAULT_CANDIDATE_MULTIPLIER, help="Oversampling factor before reranking.")
    parser.add_argument("--max-per-document", type=int, default=DEFAULT_MAX_PER_DOCUMENT, help="Maximum final chunks kept from the same document.")
    parser.add_argument("--enable-planner", dest="enable_planner", action="store_true", default=DEFAULT_ENABLE_QUERY_PLANNER, help="Enable rule-based query planning before retrieval.")
    parser.add_argument("--disable-planner", dest="enable_planner", action="store_false", help="Disable query planning and use only query variants.")
    parser.add_argument("--planner-hard-filters", dest="planner_hard_filters", action="store_true", default=DEFAULT_PLANNER_HARD_FILTERS, help="Turn planner metadata hints into hard Qdrant filters.")
    parser.add_argument("--no-planner-hard-filters", dest="planner_hard_filters", action="store_false", help="Keep planner metadata hints as soft ranking signals only.")
    parser.add_argument("--enable-llm-planner", dest="enable_llm_planner", action="store_true", default=DEFAULT_ENABLE_LLM_PLANNER, help="Use an external LLM to generate a structured retrieval plan before retrieval.")
    parser.add_argument("--disable-llm-planner", dest="enable_llm_planner", action="store_false", help="Disable external LLM planner and use only local rule-based planning.")
    parser.add_argument("--llm-planner-provider", default=DEFAULT_LLM_PLANNER_PROVIDER, choices=["openai_compatible"], help="External planner API provider.")
    parser.add_argument("--llm-planner-url", default=DEFAULT_LLM_PLANNER_URL, help="External planner API URL.")
    parser.add_argument("--llm-planner-api-key", default=DEFAULT_LLM_PLANNER_API_KEY, help="External planner API key.")
    parser.add_argument("--llm-planner-model", default=DEFAULT_LLM_PLANNER_MODEL, help="External planner model name.")
    parser.add_argument("--llm-planner-timeout", type=float, default=DEFAULT_LLM_PLANNER_TIMEOUT, help="HTTP timeout for external planner API.")
    parser.add_argument("--llm-planner-max-queries", type=int, default=DEFAULT_LLM_PLANNER_MAX_QUERIES, help="Maximum search queries preserved after merging external planner output.")
    parser.add_argument("--hnsw-ef", type=int, default=DEFAULT_QUERY_HNSW_EF, help="HNSW ef value for dense search.")
    parser.add_argument("--exact", dest="exact", action="store_true", default=DEFAULT_QUERY_EXACT, help="Use exact dense search for accuracy baselines.")
    parser.add_argument("--no-exact", dest="exact", action="store_false", help="Use approximate dense search.")
    parser.add_argument("--quantization-rescore", dest="quantization_rescore", action="store_true", default=DEFAULT_QUERY_QUANTIZATION_RESCORE, help="Rescore dense quantized candidates for better precision.")
    parser.add_argument("--no-quantization-rescore", dest="quantization_rescore", action="store_false", help="Skip dense quantization rescore for lower latency.")
    parser.add_argument("--quantization-oversampling", type=float, default=DEFAULT_QUERY_QUANTIZATION_OVERSAMPLING, help="Oversampling factor used before dense quantization rescore.")
    parser.add_argument("--intent", choices=["auto", "general", "policy", "technical"], default="auto", help="Force or infer intent.")
    parser.add_argument("--retrieval-mode", choices=["hierarchical", "flat"], default="hierarchical", help="Use parent->child retrieval or flat chunk retrieval.")
    parser.add_argument("--parallel-hierarchical", dest="parallel_hierarchical", action="store_true", default=DEFAULT_PARALLEL_HIERARCHICAL_RETRIEVAL, help="Run parent-gated and flat-child retrieval branches in parallel before merging.")
    parser.add_argument("--no-parallel-hierarchical", dest="parallel_hierarchical", action="store_false", help="Use the legacy sequential hierarchical fallback.")
    parser.add_argument("--enable-sparse-retrieval", dest="enable_sparse_retrieval", action="store_true", default=DEFAULT_ENABLE_SPARSE_RETRIEVAL, help="Enable sparse lexical retrieval and merge it with dense search.")
    parser.add_argument("--disable-sparse-retrieval", dest="enable_sparse_retrieval", action="store_false", help="Disable sparse lexical retrieval.")
    parser.add_argument("--bge-m3-model-path", default=str(DEFAULT_BGE_M3_MODEL_PATH), help="Local BGE-M3 model path used for optional dense+sparse retrieval.")
    parser.add_argument("--bge-m3-device", default=DEFAULT_BGE_M3_DEVICE, help="BGE-M3 device, e.g. auto/cpu/cuda:0.")
    parser.add_argument("--bge-m3-batch-size", type=int, default=DEFAULT_BGE_M3_BATCH_SIZE, help="BGE-M3 query batch size.")
    parser.add_argument("--bge-m3-query-max-length", type=int, default=DEFAULT_BGE_M3_QUERY_MAX_LENGTH, help="BGE-M3 query tokenizer max length.")
    parser.add_argument("--bge-m3-passage-max-length", type=int, default=DEFAULT_BGE_M3_PASSAGE_MAX_LENGTH, help="BGE-M3 passage tokenizer max length.")
    parser.add_argument("--bge-m3-use-fp16", dest="bge_m3_use_fp16", action="store_true", default=DEFAULT_BGE_M3_USE_FP16, help="Enable fp16 when loading BGE-M3 on CUDA.")
    parser.add_argument("--no-bge-m3-use-fp16", dest="bge_m3_use_fp16", action="store_false", help="Disable fp16 for BGE-M3.")
    parser.add_argument("--enable-bge-dense-retrieval", dest="enable_bge_dense_retrieval", action="store_true", default=DEFAULT_ENABLE_BGE_DENSE_RETRIEVAL, help="Enable BGE-M3 dense retrieval and merge it with the existing retrievers.")
    parser.add_argument("--disable-bge-dense-retrieval", dest="enable_bge_dense_retrieval", action="store_false", help="Disable BGE-M3 dense retrieval.")
    parser.add_argument("--enable-bge-sparse-retrieval", dest="enable_bge_sparse_retrieval", action="store_true", default=DEFAULT_ENABLE_BGE_SPARSE_RETRIEVAL, help="Enable BGE-M3 learned sparse retrieval and merge it with the existing retrievers.")
    parser.add_argument("--disable-bge-sparse-retrieval", dest="enable_bge_sparse_retrieval", action="store_false", help="Disable BGE-M3 learned sparse retrieval.")

    parser.add_argument("--enable-api-rerank", dest="enable_api_rerank", action="store_true", default=DEFAULT_ENABLE_API_RERANK, help="Enable external rerank API.")
    parser.add_argument("--disable-api-rerank", dest="enable_api_rerank", action="store_false", help="Disable external rerank API.")
    parser.add_argument("--rerank-provider", default=DEFAULT_RERANK_PROVIDER, choices=["cohere", "jina", "dashscope"], help="External rerank API provider.")
    parser.add_argument("--rerank-url", default=DEFAULT_RERANK_URL, help="External rerank API URL.")
    parser.add_argument("--rerank-api-key", default=DEFAULT_RERANK_API_KEY, help="External rerank API key.")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL, help="External rerank model name.")
    parser.add_argument("--rerank-top-n", type=int, default=DEFAULT_RERANK_TOP_N, help="How many reranked candidates to return from API.")
    parser.add_argument("--rerank-max-docs", type=int, default=DEFAULT_RERANK_MAX_DOCS, help="Maximum number of retrieved candidates sent to rerank API.")
    parser.add_argument("--rerank-max-chars-per-doc", type=int, default=DEFAULT_RERANK_MAX_CHARS_PER_DOC, help="Maximum chars per candidate sent to rerank API.")
    parser.add_argument("--rerank-timeout", type=float, default=DEFAULT_RERANK_TIMEOUT, help="HTTP timeout for rerank API.")
    parser.add_argument("--enable-local-rerank", action="store_true", default=DEFAULT_ENABLE_LOCAL_RERANK, help="Enable a local sentence-transformers CrossEncoder reranker.")
    parser.add_argument("--local-rerank-model-path", default=DEFAULT_LOCAL_RERANK_MODEL_PATH, help="Local cross-encoder reranker model path.")
    parser.add_argument("--local-rerank-max-docs", type=int, default=DEFAULT_LOCAL_RERANK_MAX_DOCS, help="Maximum candidates scored by local reranker.")
    parser.add_argument("--local-rerank-batch-size", type=int, default=DEFAULT_LOCAL_RERANK_BATCH_SIZE, help="Local reranker batch size.")
    parser.add_argument("--answer-mode", choices=["none", "grounded"], default=DEFAULT_ANSWER_MODE, help="Whether to synthesize a grounded answer from selected evidence.")
    parser.add_argument("--show-evidence", dest="show_evidence", action="store_true", default=DEFAULT_ANSWER_SHOW_EVIDENCE, help="Append local evidence sources after the final answer.")
    parser.add_argument("--hide-evidence", dest="show_evidence", action="store_false", help="Do not append local evidence sources after the final answer.")
    parser.add_argument("--answer-evidence-top-k", type=int, default=DEFAULT_ANSWER_EVIDENCE_TOP_K, help="Maximum evidence sources shown in answer-only mode.")
    parser.add_argument("--evidence-top-k", type=int, default=DEFAULT_EVIDENCE_TOP_K, help="How many evidence chunks to keep after rerank.")
    parser.add_argument("--min-evidence", type=int, default=DEFAULT_MIN_EVIDENCE, help="Minimum number of strong evidence chunks required before answering.")
    parser.add_argument("--min-evidence-score", type=float, default=DEFAULT_MIN_EVIDENCE_SCORE, help="Minimum evidence score threshold for supported evidence.")
    parser.add_argument("--max-answer-claims", type=int, default=DEFAULT_MAX_ANSWER_CLAIMS, help="Maximum number of answer claims emitted in grounded mode.")
    parser.add_argument("--core-evidence-top-k", type=int, default=DEFAULT_CORE_EVIDENCE_TOP_K, help="How many top evidence items are treated as core evidence for grounded generation.")
    parser.add_argument("--support-evidence-top-k", type=int, default=DEFAULT_SUPPORT_EVIDENCE_TOP_K, help="Maximum evidence items passed into grounded generation after core evidence.")
    parser.add_argument("--llm-context-max-tokens", type=int, default=DEFAULT_LLM_CONTEXT_MAX_TOKENS, help="Approximate token budget for evidence context sent to LLM generation/review.")
    parser.add_argument("--llm-context-max-tokens-per-evidence", type=int, default=DEFAULT_LLM_CONTEXT_MAX_TOKENS_PER_EVIDENCE, help="Approximate token cap for each evidence item after query-focused compression.")
    parser.add_argument("--llm-context-dedup-threshold", type=float, default=DEFAULT_LLM_CONTEXT_DEDUP_THRESHOLD, help="Similarity threshold used to drop near-duplicate evidence in LLM context.")
    parser.add_argument("--enable-llm-synthesis", dest="enable_llm_synthesis", action="store_true", default=DEFAULT_ENABLE_LLM_SYNTHESIS, help="Use an external LLM to produce a grounded final answer when evidence is sufficient.")
    parser.add_argument("--disable-llm-synthesis", dest="enable_llm_synthesis", action="store_false", help="Disable external LLM grounded generation and keep rule-based synthesis only.")
    parser.add_argument("--llm-synthesis-provider", default=DEFAULT_LLM_SYNTHESIS_PROVIDER, choices=["openai_compatible"], help="External grounded synthesis API provider.")
    parser.add_argument("--llm-synthesis-url", default=DEFAULT_LLM_SYNTHESIS_URL, help="External grounded synthesis API URL.")
    parser.add_argument("--llm-synthesis-api-key", default=DEFAULT_LLM_SYNTHESIS_API_KEY, help="External grounded synthesis API key.")
    parser.add_argument("--llm-synthesis-model", default=DEFAULT_LLM_SYNTHESIS_MODEL, help="External grounded synthesis model name.")
    parser.add_argument("--llm-synthesis-timeout", type=float, default=DEFAULT_LLM_SYNTHESIS_TIMEOUT, help="HTTP timeout for grounded synthesis API.")
    parser.add_argument("--llm-synthesis-disable-thinking", dest="llm_synthesis_disable_thinking", action="store_true", default=DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING, help="Send enable_thinking=false for the synthesis model.")
    parser.add_argument("--llm-synthesis-enable-thinking", dest="llm_synthesis_disable_thinking", action="store_false", help="Do not send enable_thinking=false for the synthesis model.")
    parser.add_argument("--enable-answer-review", dest="enable_answer_review", action="store_true", default=DEFAULT_ENABLE_ANSWER_REVIEW, help="Review the generated grounded answer and revise or refuse when it overreaches.")
    parser.add_argument("--disable-answer-review", dest="enable_answer_review", action="store_false", help="Disable post-answer review and keep the first grounded answer.")
    parser.add_argument("--enable-llm-answer-review", dest="enable_llm_answer_review", action="store_true", default=DEFAULT_ENABLE_LLM_ANSWER_REVIEW, help="Use an external LLM to review grounded answers after synthesis.")
    parser.add_argument("--disable-llm-answer-review", dest="enable_llm_answer_review", action="store_false", help="Disable LLM answer review and keep only heuristic review checks.")
    parser.add_argument("--llm-answer-review-provider", default=DEFAULT_LLM_ANSWER_REVIEW_PROVIDER, choices=["openai_compatible"], help="External answer review API provider.")
    parser.add_argument("--llm-answer-review-url", default=DEFAULT_LLM_ANSWER_REVIEW_URL, help="External answer review API URL.")
    parser.add_argument("--llm-answer-review-api-key", default=DEFAULT_LLM_ANSWER_REVIEW_API_KEY, help="External answer review API key.")
    parser.add_argument("--llm-answer-review-model", default=DEFAULT_LLM_ANSWER_REVIEW_MODEL, help="External answer review model name.")
    parser.add_argument("--llm-answer-review-timeout", type=float, default=DEFAULT_LLM_ANSWER_REVIEW_TIMEOUT, help="HTTP timeout for answer review API.")
    parser.add_argument("--enable-reflection", dest="enable_reflection", action="store_true", default=DEFAULT_ENABLE_REFLECTION, help="Enable reflection-driven follow-up retrieval when evidence is incomplete.")
    parser.add_argument("--disable-reflection", dest="enable_reflection", action="store_false", help="Disable reflection-driven follow-up retrieval.")
    parser.add_argument("--reflection-max-hops", type=int, default=DEFAULT_REFLECTION_MAX_HOPS, help="Maximum total retrieval hops including the first hop.")
    parser.add_argument("--reflection-overlap-threshold", type=float, default=DEFAULT_REFLECTION_OVERLAP_THRESHOLD, help="Stop re-retrieval when evidence overlap exceeds this threshold.")
    parser.add_argument("--enable-llm-reflection", dest="enable_llm_reflection", action="store_true", default=DEFAULT_ENABLE_LLM_REFLECTION, help="Use an external LLM to decide whether another retrieval hop is needed.")
    parser.add_argument("--disable-llm-reflection", dest="enable_llm_reflection", action="store_false", help="Disable LLM reflection and use heuristic reflection only.")
    parser.add_argument("--llm-reflection-provider", default=DEFAULT_LLM_REFLECTION_PROVIDER, choices=["openai_compatible"], help="External reflection API provider.")
    parser.add_argument("--llm-reflection-url", default=DEFAULT_LLM_REFLECTION_URL, help="External reflection API URL.")
    parser.add_argument("--llm-reflection-api-key", default=DEFAULT_LLM_REFLECTION_API_KEY, help="External reflection API key.")
    parser.add_argument("--llm-reflection-model", default=DEFAULT_LLM_REFLECTION_MODEL, help="External reflection model name.")
    parser.add_argument("--llm-reflection-timeout", type=float, default=DEFAULT_LLM_REFLECTION_TIMEOUT, help="HTTP timeout for reflection API.")
    parser.add_argument("--session-id", default="", help="Optional conversation session id used for memory and trace grouping.")
    parser.add_argument("--enable-memory", dest="enable_memory", action="store_true", default=DEFAULT_ENABLE_MEMORY, help="Enable conversation memory persistence for session-based follow-up questions.")
    parser.add_argument("--disable-memory", dest="enable_memory", action="store_false", help="Disable conversation memory even when session-id is provided.")
    parser.add_argument("--enable-contextualizer", dest="enable_contextualizer", action="store_true", default=DEFAULT_ENABLE_CONTEXTUALIZER, help="Rewrite follow-up questions into standalone retrieval queries using memory.")
    parser.add_argument("--disable-contextualizer", dest="enable_contextualizer", action="store_false", help="Disable query contextualization and use the raw user query directly.")
    parser.add_argument("--memory-store-dir", default=str(DEFAULT_MEMORY_STORE_DIR), help="Directory used to persist session memory JSON files.")
    parser.add_argument("--memory-max-turns", type=int, default=DEFAULT_MEMORY_MAX_TURNS, help="Maximum number of recent turns retained per session before history is compressed.")
    parser.add_argument("--context-history-turns", type=int, default=DEFAULT_CONTEXT_HISTORY_TURNS, help="How many recent turns are sent to the contextualizer.")
    parser.add_argument("--enable-trace", dest="enable_trace", action="store_true", default=DEFAULT_TRACE_ENABLED, help="Write per-turn trace JSON files for observability and evaluation.")
    parser.add_argument("--disable-trace", dest="enable_trace", action="store_false", help="Disable trace logging.")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR), help="Directory used to store search traces.")
    parser.add_argument("--trace-top-k", type=int, default=DEFAULT_TRACE_TOP_K, help="How many retrieved/reranked items to keep in each trace hop.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed retrieval diagnostics.")
    return parser


def hydrate_search_candidate(item: Dict[str, Any]) -> SearchCandidate:
    return SearchCandidate(
        id=item["id"],
        chunk_uid=item["chunk_uid"],
        chunk_level=item["chunk_level"],
        parent_chunk_uid=item["parent_chunk_uid"],
        source_name=item["source_name"],
        semantic_score=item["semantic_score"],
        doc_title=item["doc_title"],
        source_file=item["source_file"],
        section_title=item["section_title"],
        header_path=item.get("header_path", []),
        chunk_type=item["chunk_type"],
        quality_score=item["quality_score"],
        text_preview=item["text_preview"],
        text=item["text"],
        payload=item,
        matched_queries=item.get("matched_queries", []),
        score_breakdown=item.get("score_breakdown", {}),
    )


def hydrate_query_plan(item: Dict[str, Any], query: str, intent: str) -> QueryPlan:
    return QueryPlan(
        original_query=item.get("original_query", query),
        intent=item.get("intent", intent),
        task_type=item.get("task_type", classify_query_task(query, item.get("intent", intent))),
        normalized_query=item.get("normalized_query", normalize_text(query)),
        theme_terms=item.get("theme_terms", []),
        entity_terms=item.get("entity_terms", []),
        constraint_terms=item.get("constraint_terms", []),
        time_terms=item.get("time_terms", []),
        evidence_focus=item.get("evidence_focus", []),
        filter_hints=item.get("filter_hints", {}),
        sub_queries=item.get("sub_queries", []),
        search_queries=item.get("search_queries", []),
        notes=item.get("notes", []),
        needs_multi_hop=bool(item.get("needs_multi_hop", False)),
    )


def hydrate_evidence_item(item: Dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        rank=item["rank"],
        chunk_uid=item["chunk_uid"],
        source_file=item["source_file"],
        doc_title=item["doc_title"],
        section_title=item["section_title"],
        chunk_level=item["chunk_level"],
        group=item["group"],
        quote=item["quote"],
        evidence_score=item["evidence_score"],
        body_support_score=item["body_support_score"],
        answerability_score=item["answerability_score"],
        final_score=item["final_score"],
        citation=item.get("citation", {}),
        tier=item.get("tier", ""),
    )


def hydrate_answer(item: Optional[Dict[str, Any]]) -> Optional[AnswerSynthesis]:
    if not item:
        return None
    return AnswerSynthesis(
        status=item["status"],
        confidence=item["confidence"],
        answer=item["answer"],
        claims=item.get("claims", []),
        conflicts=item.get("conflicts", []),
        refusal_reason=item.get("refusal_reason", ""),
        citations=item.get("citations", []),
        gaps=item.get("gaps", []),
        followups=item.get("followups", []),
        grounding_mode=item.get("grounding_mode", "extractive"),
        llm_model=item.get("llm_model", ""),
        review_status=item.get("review_status", ""),
        review_issues=item.get("review_issues", []),
    )


def extract_json_payload(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if isinstance(text, list):
        for item in text:
            if isinstance(item, dict):
                return item
        raise RuntimeError("Planner response list does not contain a JSON object.")

    content = str(text or "").strip()
    if not content:
        raise RuntimeError("Planner response is empty.")

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, flags=re.S | re.I)
    if fence_match:
        content = fence_match.group(1).strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        snippet = content[start : end + 1]
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError("Planner response did not contain a valid JSON object.")


def call_external_planner_api(
    query: str,
    intent: str,
    rule_plan: QueryPlan,
    provider: str,
    api_url: str,
    api_key: str,
    model: str,
    timeout: float,
) -> Dict[str, Any]:
    if not api_url or not api_key or not model:
        raise RuntimeError("External planner requires URL, API key, and model.")

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("External planner requires the optional 'requests' package.") from exc

    provider = normalize_text(provider)
    if provider != "openai_compatible":
        raise RuntimeError(f"Unsupported planner provider: {provider}")

    system_prompt = (
        "你是中文行研 RAG 智能体的检索规划节点。"
        "只返回一个 JSON 对象，不要回答用户问题。"
        "你的任务是判断问题类型，抽取可投/产业实体与约束条件，并生成高质量检索查询。"
        "检索查询应在相关时覆盖市场规模、增速、供需、竞争格局、价值链、商业模式、盈利能力、政策、风险、数据来源和时间口径。\n"
        "task_type 只能取：market、comparison、definition、trend、procedure、root_cause、status、fact。\n"
        "intent 只能取：general、policy、technical。\n"
        "返回字段必须包括：intent、task_type、entity_terms、constraint_terms、time_terms、evidence_focus、"
        "search_queries、sub_queries、filter_hints、notes、needs_multi_hop。\n"
        "filter_hints 可以包含：doc_type、chunk_type、negative_doc_type、negative_chunk_type。\n"
        "search_queries 应短小、可检索，通常 6 到 12 条，并尽量覆盖不同角度：市场规模、增速、企业对比、政策、技术、财务、风险与时间口径。"
        "如果问题是行研、竞争分析、投融资或经营分析，尽量拆成多个互补查询，不要只保留单一角度。"
        "除非用户明确询问公司定位，否则不要生成偏宣传口径的查询。"
    )
    user_payload = {
        "query": query,
        "rule_plan": rule_plan.to_dict(),
        "current_intent": intent,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }
    data = post_external_json(
        api_url=normalize_openai_compatible_chat_url(api_url),
        headers=headers,
        payload=payload,
        timeout=timeout,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Planner response did not include choices.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(item))
        content = "\n".join(text_parts)
    return extract_json_payload(content)


def merge_filter_hints(primary: Dict[str, List[str]], secondary: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for source in [primary or {}, secondary or {}]:
        for key, values in source.items():
            existing = merged.get(key, [])
            merged[key] = unique_preserve(existing + [str(value).strip() for value in values if str(value).strip()], max_items=12)
    return merged


def merge_external_query_plan(
    rule_plan: QueryPlan,
    planner_payload: Dict[str, Any],
    max_search_queries: int,
) -> QueryPlan:
    valid_task_types = set(_TASK_DOC_TYPE_HINTS.keys())
    valid_intents = {"general", "policy", "technical"}

    external_intent = normalize_text(planner_payload.get("intent", ""))
    external_task_type = normalize_text(planner_payload.get("task_type", ""))
    merged_intent = external_intent if external_intent in valid_intents else rule_plan.intent
    merged_task_type = external_task_type if external_task_type in valid_task_types else rule_plan.task_type

    merged_plan = QueryPlan(
        original_query=rule_plan.original_query,
        intent=merged_intent,
        task_type=merged_task_type,
        normalized_query=rule_plan.normalized_query,
        theme_terms=unique_preserve(list(planner_payload.get("theme_terms", [])) + rule_plan.theme_terms, max_items=10),
        entity_terms=unique_preserve(list(planner_payload.get("entity_terms", [])) + rule_plan.entity_terms, max_items=10),
        constraint_terms=unique_preserve(list(planner_payload.get("constraint_terms", [])) + rule_plan.constraint_terms, max_items=10),
        time_terms=unique_preserve(list(planner_payload.get("time_terms", [])) + rule_plan.time_terms, max_items=6),
        evidence_focus=unique_preserve(list(planner_payload.get("evidence_focus", [])) + rule_plan.evidence_focus, max_items=8),
        filter_hints=merge_filter_hints(rule_plan.filter_hints, planner_payload.get("filter_hints", {})),
        sub_queries=unique_preserve(list(planner_payload.get("sub_queries", [])) + rule_plan.sub_queries, max_items=8),
        search_queries=unique_preserve(list(planner_payload.get("search_queries", [])) + rule_plan.search_queries, max_items=max_search_queries),
        notes=unique_preserve(rule_plan.notes + list(planner_payload.get("notes", [])) + ["planner_source=external_llm"], max_items=16),
        needs_multi_hop=bool(planner_payload.get("needs_multi_hop", rule_plan.needs_multi_hop)),
    )
    if not merged_plan.evidence_focus:
        merged_plan.evidence_focus = infer_evidence_focus(merged_plan.original_query, merged_plan.intent, merged_plan.task_type)
    if not merged_plan.filter_hints:
        merged_plan.filter_hints = infer_filter_hints(merged_plan.original_query, merged_plan.intent, merged_plan.task_type)
    if not merged_plan.search_queries:
        merged_plan.search_queries = rule_plan.search_queries
    return merged_plan


def maybe_apply_external_planner(args, query: str, intent: str, rule_plan: QueryPlan) -> tuple[QueryPlan, Dict[str, Any]]:
    if not getattr(args, "enable_llm_planner", False):
        return rule_plan, {}
    if not args.llm_planner_url or not args.llm_planner_api_key or not args.llm_planner_model:
        return rule_plan, {}
    try:
        planner_payload = call_external_planner_api(
            query=query,
            intent=intent,
            rule_plan=rule_plan,
            provider=args.llm_planner_provider,
            api_url=args.llm_planner_url,
            api_key=args.llm_planner_api_key,
            model=args.llm_planner_model,
            timeout=args.llm_planner_timeout,
        )
    except Exception as exc:
        warning = f"[planner-warning] external planner fallback to local rules: {exc}"
        print(warning, file=sys.stderr)
        return rule_plan, {
            "type": "planner",
            "error": str(exc),
            "request": {
                "query": query,
                "intent": intent,
                "rule_plan": rule_plan.to_dict(),
            },
        }
    merged_plan = merge_external_query_plan(
        rule_plan=rule_plan,
        planner_payload=planner_payload,
        max_search_queries=max(args.llm_planner_max_queries, args.query_variants * 2),
    )
    return merged_plan, {
        "type": "planner",
        "request": {
            "query": query,
            "intent": intent,
            "rule_plan": rule_plan.to_dict(),
        },
        "response": planner_payload,
    }


def merge_followup_plan(base_plan: QueryPlan, followup_query: str, intent: str, max_variants: int) -> QueryPlan:
    followup_plan = build_query_plan(followup_query, intent, max_variants=max_variants)
    merged_intent = base_plan.intent or followup_plan.intent
    merged_task_type = base_plan.task_type or followup_plan.task_type
    return QueryPlan(
        original_query=base_plan.original_query,
        intent=merged_intent,
        task_type=merged_task_type,
        normalized_query=base_plan.normalized_query,
        theme_terms=unique_preserve(followup_plan.theme_terms + base_plan.theme_terms, max_items=12),
        entity_terms=unique_preserve(base_plan.entity_terms + followup_plan.entity_terms, max_items=12),
        constraint_terms=unique_preserve(base_plan.constraint_terms + followup_plan.constraint_terms, max_items=12),
        time_terms=unique_preserve(base_plan.time_terms + followup_plan.time_terms, max_items=8),
        evidence_focus=unique_preserve(base_plan.evidence_focus + followup_plan.evidence_focus, max_items=10),
        filter_hints=merge_filter_hints(base_plan.filter_hints, followup_plan.filter_hints),
        sub_queries=unique_preserve(base_plan.sub_queries + [followup_query] + followup_plan.sub_queries, max_items=12),
        search_queries=unique_preserve(followup_plan.search_queries + [followup_query], max_items=max(max_variants * 3, 8)),
        notes=unique_preserve(
            base_plan.notes + [f"reflection_query={followup_query}"] + followup_plan.notes,
            max_items=20,
        ),
        needs_multi_hop=True,
    )


def add_timing_value(timings: Dict[str, float], key: str, value: float) -> None:
    timings[key] = round(float(timings.get(key, 0.0) or 0.0) + float(value or 0.0), 2)


def execute_retrieval_pass(
    *,
    args,
    client: Any,
    qmodels: Any,
    qwen_embedder: Any,
    bge_helper: Any = None,
    active_query: str,
    active_plan: QueryPlan,
    intent: str,
    enable_sparse_retrieval: bool,
    enable_bge_dense_retrieval: bool,
    enable_bge_sparse_retrieval: bool,
) -> Dict[str, Any]:
    hop_timings: Dict[str, float] = {}
    query_variant_limit = max(1, args.query_variants)
    if args.enable_planner:
        query_variant_limit = max(
            query_variant_limit,
            min(max(args.llm_planner_max_queries, args.query_variants * 2), max(4, len(active_plan.search_queries))),
        )
    query_variants = unique_preserve(
        active_plan.search_queries or expand_query_variants(active_query, args.query_variants),
        max_items=query_variant_limit,
    )
    instruction_queries = [build_instruction_query(value, intent, args.disable_instruction) for value in query_variants]

    embedding_started = time.perf_counter()
    query_vectors = qwen_embedder.encode(
        list(instruction_queries),
        batch_size=min(8, max(1, len(instruction_queries))),
    )
    hop_timings["embedding_ms"] = round((time.perf_counter() - embedding_started) * 1000, 2)

    sparse_started = time.perf_counter()
    sparse_query_vectors = []
    if enable_sparse_retrieval:
        for variant in query_variants:
            sparse_payload = build_sparse_vector(variant)
            sparse_query_vectors.extend(build_qdrant_sparse_vectors(qmodels, [sparse_payload]))
    hop_timings["sparse_query_ms"] = round((time.perf_counter() - sparse_started) * 1000, 2)

    bge_query_vectors: List[Any] = []
    bge_sparse_query_vectors: List[Any] = []
    bge_started = time.perf_counter()
    if bge_helper is not None and (enable_bge_dense_retrieval or enable_bge_sparse_retrieval):
        bge_outputs = bge_helper.encode_queries(
            list(query_variants),
            batch_size=min(max(1, int(args.bge_m3_batch_size)), max(1, len(query_variants))),
            return_dense=enable_bge_dense_retrieval,
            return_sparse=enable_bge_sparse_retrieval,
        )
        if enable_bge_dense_retrieval:
            bge_query_vectors = list(bge_outputs.get("dense_vecs", []) or [])
        if enable_bge_sparse_retrieval:
            bge_sparse_query_vectors = build_qdrant_sparse_vectors(
                qmodels,
                list(bge_outputs.get("sparse_vectors", []) or []),
            )
    hop_timings["bge_query_ms"] = round((time.perf_counter() - bge_started) * 1000, 2)

    query_filter = build_filter(args, qmodels, plan=active_plan)
    dense_search_params = build_search_params(
        qmodels=qmodels,
        hnsw_ef=args.hnsw_ef,
        exact=args.exact,
        quantization_rescore=args.quantization_rescore,
        quantization_oversampling=args.quantization_oversampling,
    )
    per_query_limit = max(max(1, args.top_k), args.top_k * args.candidate_multiplier, args.evidence_top_k * 2)
    use_hierarchical = args.retrieval_mode == "hierarchical" and not args.chunk_level and not args.chunk_uid

    def run_retrieval_branch(
        *,
        dense_query_vectors: Sequence[Any],
        sparse_query_vectors: Sequence[Any],
        dense_source_name: str,
        dense_score_key: str,
        dense_vector_name: str,
        sparse_source_name: str,
        sparse_score_key: str,
        sparse_vector_name: str,
    ) -> List[SearchCandidate]:
        branch_merged: List[SearchCandidate] = []
        if use_hierarchical and dense_query_vectors:
            if args.parallel_hierarchical:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    hierarchical_future = executor.submit(
                        hierarchical_dense_search,
                        client=client,
                        qmodels=qmodels,
                        collection=args.collection,
                        query_variants=query_variants,
                        query_vectors=dense_query_vectors,
                        query_sparse_vectors=sparse_query_vectors,
                        base_filter=query_filter,
                        search_params=dense_search_params,
                        score_threshold=args.score_threshold,
                        limit=per_query_limit,
                        preview_chars=args.preview_chars,
                        source_name=dense_source_name,
                        score_key=dense_score_key,
                        dense_vector_name=dense_vector_name,
                        sparse_source_name=sparse_source_name,
                        sparse_score_key=sparse_score_key,
                        sparse_vector_name=sparse_vector_name,
                    )
                    flat_child_future = executor.submit(
                        flat_child_search,
                        client=client,
                        qmodels=qmodels,
                        collection=args.collection,
                        query_variants=query_variants,
                        query_vectors=dense_query_vectors,
                        query_sparse_vectors=sparse_query_vectors,
                        base_filter=query_filter,
                        search_params=dense_search_params,
                        score_threshold=args.score_threshold,
                        limit=per_query_limit,
                        preview_chars=args.preview_chars,
                        dense_source_name=f"{dense_source_name}_flat_child",
                        dense_score_key=f"{dense_score_key}_flat_child",
                        dense_vector_name=dense_vector_name,
                        sparse_source_name=f"{sparse_source_name}_flat_child",
                        sparse_score_key=f"{sparse_score_key}_flat_child",
                        sparse_vector_name=sparse_vector_name,
                    )
                    parent_candidates, child_candidates = hierarchical_future.result()
                    flat_child_candidates = flat_child_future.result()
                branch_merged = merge_candidates(child_candidates, flat_child_candidates)
                if not branch_merged:
                    branch_merged = parent_candidates
            else:
                parent_candidates, child_candidates = hierarchical_dense_search(
                    client=client,
                    qmodels=qmodels,
                    collection=args.collection,
                    query_variants=query_variants,
                    query_vectors=dense_query_vectors,
                    query_sparse_vectors=sparse_query_vectors,
                    base_filter=query_filter,
                    search_params=dense_search_params,
                    score_threshold=args.score_threshold,
                    limit=per_query_limit,
                    preview_chars=args.preview_chars,
                    source_name=dense_source_name,
                    score_key=dense_score_key,
                    dense_vector_name=dense_vector_name,
                    sparse_source_name=sparse_source_name,
                    sparse_score_key=sparse_score_key,
                    sparse_vector_name=sparse_vector_name,
                )
                branch_merged = child_candidates or parent_candidates
                if needs_flat_child_fallback(active_query, child_candidates):
                    flat_child_candidates = flat_child_search(
                        client=client,
                        qmodels=qmodels,
                        collection=args.collection,
                        query_variants=query_variants,
                        query_vectors=dense_query_vectors,
                        query_sparse_vectors=sparse_query_vectors,
                        base_filter=query_filter,
                        search_params=dense_search_params,
                        score_threshold=args.score_threshold,
                        limit=per_query_limit,
                        preview_chars=args.preview_chars,
                        dense_source_name=f"{dense_source_name}_flat_child",
                        dense_score_key=f"{dense_score_key}_flat_child",
                        dense_vector_name=dense_vector_name,
                        sparse_source_name=f"{sparse_source_name}_flat_child",
                        sparse_score_key=f"{sparse_score_key}_flat_child",
                        sparse_vector_name=sparse_vector_name,
                    )
                    branch_merged = merge_candidates(child_candidates, flat_child_candidates) if child_candidates else flat_child_candidates
            if not branch_merged:
                dense_candidates = dense_search(
                    client=client,
                    collection=args.collection,
                    query_variants=query_variants,
                    query_vectors=dense_query_vectors,
                    query_filter=query_filter,
                    search_params=dense_search_params,
                    score_threshold=args.score_threshold,
                    limit=per_query_limit,
                    preview_chars=args.preview_chars,
                    source_name=dense_source_name,
                    score_key=dense_score_key,
                    vector_name=dense_vector_name,
                )
                sparse_candidates = []
                if sparse_query_vectors:
                    sparse_candidates = sparse_search(
                        client=client,
                        collection=args.collection,
                        query_variants=query_variants,
                        query_sparse_vectors=sparse_query_vectors,
                        query_filter=query_filter,
                        limit=per_query_limit,
                        preview_chars=args.preview_chars,
                        source_name=sparse_source_name,
                        score_key=sparse_score_key,
                        vector_name=sparse_vector_name,
                    )
                branch_merged = merge_candidates(dense_candidates, sparse_candidates)
        else:
            dense_candidates = []
            if dense_query_vectors:
                dense_candidates = dense_search(
                    client=client,
                    collection=args.collection,
                    query_variants=query_variants,
                    query_vectors=dense_query_vectors,
                    query_filter=query_filter,
                    search_params=dense_search_params,
                    score_threshold=args.score_threshold,
                    limit=per_query_limit,
                    preview_chars=args.preview_chars,
                    source_name=dense_source_name,
                    score_key=dense_score_key,
                    vector_name=dense_vector_name,
                )
            sparse_candidates = []
            if sparse_query_vectors:
                sparse_candidates = sparse_search(
                    client=client,
                    collection=args.collection,
                    query_variants=query_variants,
                    query_sparse_vectors=sparse_query_vectors,
                    query_filter=query_filter,
                    limit=per_query_limit,
                    preview_chars=args.preview_chars,
                    source_name=sparse_source_name,
                    score_key=sparse_score_key,
                    vector_name=sparse_vector_name,
                )
            branch_merged = merge_candidates(dense_candidates, sparse_candidates)
        return branch_merged

    retrieval_started = time.perf_counter()
    merged = run_retrieval_branch(
        dense_query_vectors=query_vectors,
        sparse_query_vectors=sparse_query_vectors,
        dense_source_name="qwen_dense",
        dense_score_key="qwen_dense",
        dense_vector_name=DEFAULT_DENSE_VECTOR_NAME,
        sparse_source_name="lexical_sparse",
        sparse_score_key="lexical_sparse",
        sparse_vector_name=DEFAULT_SPARSE_VECTOR_NAME,
    )
    if bge_query_vectors or bge_sparse_query_vectors:
        merged = merge_candidates(
            merged,
            run_retrieval_branch(
                dense_query_vectors=bge_query_vectors,
                sparse_query_vectors=bge_sparse_query_vectors,
                dense_source_name="bge_dense",
                dense_score_key="bge_dense",
                dense_vector_name=DEFAULT_BGE_DENSE_VECTOR_NAME,
                sparse_source_name="bge_sparse",
                sparse_score_key="bge_sparse",
                sparse_vector_name=DEFAULT_BGE_SPARSE_VECTOR_NAME,
            ),
        )
    hop_timings["retrieval_ms"] = round((time.perf_counter() - retrieval_started) * 1000, 2)

    external_rerank_started = time.perf_counter()
    api_rerank_ok = False
    if args.enable_api_rerank:
        api_rerank_ok = add_external_rerank_scores(
            query=active_query,
            candidates=merged,
            provider=args.rerank_provider,
            api_url=args.rerank_url,
            api_key=args.rerank_api_key,
            model=args.rerank_model,
            max_docs=args.rerank_max_docs,
            top_n=args.rerank_top_n,
            max_chars_per_doc=args.rerank_max_chars_per_doc,
            timeout=args.rerank_timeout,
        )
    hop_timings["external_rerank_ms"] = round((time.perf_counter() - external_rerank_started) * 1000, 2)
    hop_timings["external_rerank_ok"] = bool(api_rerank_ok) if args.enable_api_rerank else False

    local_rerank_started = time.perf_counter()
    local_rerank_fallback = bool(args.enable_api_rerank and not api_rerank_ok and args.local_rerank_model_path)
    if (args.enable_local_rerank or local_rerank_fallback) and args.local_rerank_model_path:
        try:
            add_local_rerank_scores(
                query=active_query,
                candidates=merged,
                model_path=args.local_rerank_model_path,
                max_docs=args.local_rerank_max_docs,
                max_chars_per_doc=args.rerank_max_chars_per_doc,
                batch_size=args.local_rerank_batch_size,
            )
        except Exception as exc:
            hop_timings["local_rerank_error"] = str(exc)[:300]
    hop_timings["local_cross_encoder_ms"] = round((time.perf_counter() - local_rerank_started) * 1000, 2)
    hop_timings["local_rerank_fallback"] = local_rerank_fallback

    return {
        "query_variants": query_variants,
        "instruction_queries": instruction_queries,
        "candidates": merged,
        "timings": hop_timings,
        "active_plan": active_plan,
    }


def run_search(args, query: str) -> Dict[str, Any]:
    total_started = time.perf_counter()
    timings: Dict[str, float] = {}
    original_query = str(query or "").strip()
    session_id = str(getattr(args, "session_id", "") or "").strip()
    trace_session_id = session_id or "standalone"
    turn_id = generate_turn_id()
    llm_calls: List[Dict[str, Any]] = []
    notices: List[str] = []
    raw_synthesis_disable_thinking = getattr(
        args,
        "llm_synthesis_disable_thinking",
        DEFAULT_LLM_SYNTHESIS_DISABLE_THINKING,
    )
    synthesis_disable_thinking = (
        raw_synthesis_disable_thinking
        if isinstance(raw_synthesis_disable_thinking, bool)
        else str(raw_synthesis_disable_thinking).strip().lower() in {"1", "true", "yes", "on"}
    )

    memory = ConversationMemory(
        store_dir=Path(getattr(args, "memory_store_dir", DEFAULT_MEMORY_STORE_DIR)),
        max_turns=getattr(args, "memory_max_turns", DEFAULT_MEMORY_MAX_TURNS),
    )
    contextualization_started = time.perf_counter()
    contextualization_llm_config = {
        "provider": args.llm_synthesis_provider,
        "url": args.llm_synthesis_url,
        "api_key": args.llm_synthesis_api_key,
        "model": args.llm_synthesis_model,
        "timeout": args.llm_synthesis_timeout,
        "disable_thinking": synthesis_disable_thinking,
    }
    if session_id and getattr(args, "enable_memory", False) and getattr(args, "enable_contextualizer", False):
        contextualization = contextualize_query(
            query=original_query,
            session_id=session_id,
            memory=memory,
            llm_config=contextualization_llm_config,
            history_turns=getattr(args, "context_history_turns", DEFAULT_CONTEXT_HISTORY_TURNS),
        )
    else:
        contextualization = {
            "original_query": original_query,
            "standalone_query": original_query,
            "memory_summary": "",
            "recent_turns": [],
            "reused_chunk_uids": [],
            "source": "disabled",
            "note": "memory_or_contextualizer_disabled",
            "llm_call": {},
        }
    timings["contextualization_ms"] = round((time.perf_counter() - contextualization_started) * 1000, 2)
    if contextualization.get("llm_call"):
        llm_calls.append(contextualization["llm_call"])

    retrieval_query = str(contextualization.get("standalone_query") or original_query).strip() or original_query
    trace_recorder = TraceRecorder(
        enabled=bool(getattr(args, "enable_trace", DEFAULT_TRACE_ENABLED)),
        trace_dir=Path(getattr(args, "trace_dir", DEFAULT_TRACE_DIR)),
        session_id=trace_session_id,
        turn_id=turn_id,
        original_query=original_query,
        standalone_query=retrieval_query,
    )
    trace_recorder.set_memory(
        {
            "session_id": session_id,
            "summary": contextualization.get("memory_summary", ""),
            "recent_turns": contextualization.get("recent_turns", []),
            "reused_chunk_uids": contextualization.get("reused_chunk_uids", []),
            "source": contextualization.get("source", ""),
            "note": contextualization.get("note", ""),
        }
    )
    if contextualization.get("llm_call"):
        trace_recorder.add_llm_call(contextualization.get("llm_call"))

    planning_started = time.perf_counter()
    intent = classify_query_intent(retrieval_query) if args.intent == "auto" else args.intent
    planner_trace: Dict[str, Any] = {}
    if args.enable_planner:
        plan = build_query_plan(retrieval_query, intent, max_variants=args.query_variants)
        plan, planner_trace = maybe_apply_external_planner(args, retrieval_query, intent, plan)
        intent = plan.intent
    else:
        task_type = classify_query_task(retrieval_query, intent)
        plan = QueryPlan(
            original_query=retrieval_query,
            intent=intent,
            task_type=task_type,
            normalized_query=normalize_text(retrieval_query),
            evidence_focus=infer_evidence_focus(retrieval_query, intent, task_type),
            filter_hints=infer_filter_hints(retrieval_query, intent, task_type),
            sub_queries=[retrieval_query],
            search_queries=expand_query_variants(retrieval_query, args.query_variants),
        )
    timings["planning_ms"] = round((time.perf_counter() - planning_started) * 1000, 2)
    trace_recorder.set_plan(plan.to_dict())
    if planner_trace:
        llm_calls.append(planner_trace)
        trace_recorder.add_llm_call(planner_trace)

    _, qmodels = _require_qdrant_client()
    resolved_url = args.url
    resolved_db_path = args.db_path
    if not resolved_url:
        probed_url = probe_local_qdrant_url(args.collection)
        if probed_url:
            resolved_url = probed_url
        else:
            resolved_db_path = resolve_local_qdrant_path(args.db_path, args.collection)
    try:
        client = build_client(resolved_db_path, resolved_url, args.api_key, args.prefer_grpc)
    except RuntimeError as exc:
        if not resolved_url and is_qdrant_local_lock_error(exc):
            raise RuntimeError(build_qdrant_lock_message(resolved_db_path)) from exc
        raise

    results: List[SearchCandidate] = []
    evidence_items: List[EvidenceItem] = []
    answer: Optional[AnswerSynthesis] = None
    query_variants: List[str] = []
    instruction_queries: List[str] = []
    reflections: List[Dict[str, Any]] = []
    answer_review: Dict[str, Any] = {}
    qwen_embedder = None
    bge_helper = None
    effective_sparse_retrieval = False
    effective_bge_dense_retrieval = False
    effective_bge_sparse_retrieval = False
    try:
        try:
            collection_exists = client.collection_exists(args.collection)
        except Exception as exc:
            if resolved_url and is_qdrant_remote_connection_error(exc):
                fallback_db_path = resolve_local_qdrant_path(args.db_path, args.collection)
                if local_qdrant_collection_exists(fallback_db_path, args.collection):
                    notices.append(
                        f"Qdrant server '{resolved_url}' is unreachable; using local Qdrant storage '{fallback_db_path}' instead."
                    )
                    try:
                        client.close()
                    except Exception:
                        pass
                    resolved_url = ""
                    resolved_db_path = fallback_db_path
                    client = build_client(resolved_db_path, "", "", False)
                    collection_exists = client.collection_exists(args.collection)
                else:
                    raise RuntimeError(
                        build_qdrant_unavailable_message(args.collection, resolved_url, resolved_db_path, exc)
                    ) from exc
            else:
                raise

        if not collection_exists:
            raise RuntimeError(
                build_missing_collection_message(args.collection, resolved_url, resolved_db_path)
            )

        collection_info = client.get_collection(args.collection)
        collection_dense_vectors = extract_collection_dense_vector_names(collection_info)
        collection_sparse_vectors = extract_collection_sparse_vector_names(collection_info)
        if DEFAULT_DENSE_VECTOR_NAME not in collection_dense_vectors:
            raise RuntimeError(
                f"Qdrant collection '{args.collection}' is missing the required dense vector field '{DEFAULT_DENSE_VECTOR_NAME}'. "
                "Please re-run ingest to recreate or repair the collection."
            )

        effective_sparse_retrieval = bool(
            args.enable_sparse_retrieval and DEFAULT_SPARSE_VECTOR_NAME in collection_sparse_vectors
        )
        if args.enable_sparse_retrieval and not effective_sparse_retrieval:
            notices.append(
                f"Sparse retrieval requested but '{DEFAULT_SPARSE_VECTOR_NAME}' is not present in collection '{args.collection}', so lexical sparse retrieval was skipped."
            )

        effective_bge_dense_retrieval = bool(
            args.enable_bge_dense_retrieval
            and str(args.bge_m3_model_path or "").strip()
            and DEFAULT_BGE_DENSE_VECTOR_NAME in collection_dense_vectors
        )
        if args.enable_bge_dense_retrieval and DEFAULT_BGE_DENSE_VECTOR_NAME not in collection_dense_vectors:
            notices.append(
                f"BGE dense retrieval requested but '{DEFAULT_BGE_DENSE_VECTOR_NAME}' is not present in collection '{args.collection}'. Rebuild the collection with BGE enabled to use it."
            )

        effective_bge_sparse_retrieval = bool(
            args.enable_bge_sparse_retrieval
            and str(args.bge_m3_model_path or "").strip()
            and DEFAULT_BGE_SPARSE_VECTOR_NAME in collection_sparse_vectors
        )
        if args.enable_bge_sparse_retrieval and DEFAULT_BGE_SPARSE_VECTOR_NAME not in collection_sparse_vectors:
            notices.append(
                f"BGE sparse retrieval requested but '{DEFAULT_BGE_SPARSE_VECTOR_NAME}' is not present in collection '{args.collection}'. Rebuild the collection with BGE enabled to use it."
            )
        if (args.enable_bge_dense_retrieval or args.enable_bge_sparse_retrieval) and not str(args.bge_m3_model_path or "").strip():
            notices.append("BGE retrieval was requested but no BGE-M3 model path is configured, so BGE retrieval was skipped.")

        qwen_embedder = get_qwen_embedder(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            max_length=args.max_length,
            keep_loaded=not args.no_embedder_cache,
        )
        if effective_bge_dense_retrieval or effective_bge_sparse_retrieval:
            bge_helper = get_bge_m3_helper(
                model_path=args.bge_m3_model_path,
                device=args.bge_m3_device,
                batch_size=args.bge_m3_batch_size,
                query_max_length=args.bge_m3_query_max_length,
                passage_max_length=args.bge_m3_passage_max_length,
                use_fp16=args.bge_m3_use_fp16,
                keep_loaded=not args.no_embedder_cache,
            )
        reflection_llm_config = {
            "provider": args.llm_reflection_provider,
            "url": args.llm_reflection_url,
            "api_key": args.llm_reflection_api_key,
            "model": args.llm_reflection_model,
            "timeout": args.llm_reflection_timeout,
        }
        synthesis_config = {
            "provider": args.llm_synthesis_provider,
            "url": args.llm_synthesis_url,
            "api_key": args.llm_synthesis_api_key,
            "model": args.llm_synthesis_model,
            "timeout": args.llm_synthesis_timeout,
            "disable_thinking": synthesis_disable_thinking,
        }
        answer_review_config = {
            "provider": args.llm_answer_review_provider,
            "url": args.llm_answer_review_url,
            "api_key": args.llm_answer_review_api_key,
            "model": args.llm_answer_review_model,
            "timeout": args.llm_answer_review_timeout,
        }
        if not getattr(args, "enable_llm_synthesis", False):
            synthesis_config = {}
        if not getattr(args, "enable_llm_answer_review", False):
            answer_review_config = {}
        if not getattr(args, "enable_llm_reflection", False):
            reflection_llm_config = {}

        candidate_pool: List[SearchCandidate] = []
        accumulated_query_variants: List[str] = []
        accumulated_instruction_queries: List[str] = []
        context_engineering: Dict[str, Any] = {}
        previous_queries: List[str] = [retrieval_query]
        previous_evidence_sets: List[List[str]] = []
        current_query = retrieval_query
        max_hops = max(1, int(getattr(args, "reflection_max_hops", DEFAULT_REFLECTION_MAX_HOPS)))

        for hop_index in range(1, max_hops + 1):
            hop_query = current_query
            active_plan = plan if hop_index == 1 else merge_followup_plan(plan, hop_query, intent, args.query_variants)
            hop_output = execute_retrieval_pass(
                args=args,
                client=client,
                qmodels=qmodels,
                qwen_embedder=qwen_embedder,
                bge_helper=bge_helper,
                active_query=hop_query,
                active_plan=active_plan,
                intent=intent,
                enable_sparse_retrieval=effective_sparse_retrieval,
                enable_bge_dense_retrieval=effective_bge_dense_retrieval,
                enable_bge_sparse_retrieval=effective_bge_sparse_retrieval,
            )
            for key, value in hop_output["timings"].items():
                add_timing_value(timings, key, value)

            accumulated_query_variants = unique_preserve(
                accumulated_query_variants + list(hop_output["query_variants"]),
                max_items=max(args.query_variants * max_hops * 2, 24),
            )
            accumulated_instruction_queries = unique_preserve(
                accumulated_instruction_queries + list(hop_output["instruction_queries"]),
                max_items=max(args.query_variants * max_hops * 2, 24),
            )
            candidate_pool = merge_candidates(candidate_pool, hop_output["candidates"])

            local_scoring_started = time.perf_counter()
            reranked = rerank_candidates(
                query=retrieval_query,
                candidates=candidate_pool,
                top_k=max(args.top_k, args.evidence_top_k),
                max_per_document=max(args.max_per_document, 3),
                use_external_rerank=args.enable_api_rerank,
                plan=plan,
            )
            add_timing_value(timings, "local_rerank_ms", round((time.perf_counter() - local_scoring_started) * 1000, 2))

            results = reranked[: max(1, args.top_k)]
            evidence_started = time.perf_counter()
            evidence_items = select_evidence(
                query=retrieval_query,
                plan=plan,
                candidates=reranked,
                evidence_top_k=args.evidence_top_k,
                max_per_document=args.max_per_document,
                core_evidence_top_k=args.core_evidence_top_k,
            )
            add_timing_value(timings, "evidence_ms", round((time.perf_counter() - evidence_started) * 1000, 2))

            coverage = evaluate_evidence_coverage(plan, evidence_items)
            topic_consistency = evaluate_topic_consistency(plan, evidence_items)
            evidence_payload = [item.to_dict() for item in evidence_items]
            current_evidence_uids = evidence_chunk_uids(evidence_payload)
            answer = None
            answer_review = {}
            context_engineering = {}
            synthesis_deferred = bool(
                args.answer_mode == "grounded"
                and getattr(args, "enable_reflection", False)
                and hop_index < max_hops
                and not (
                    evidence_items
                    and coverage.get("passed", False)
                    and topic_consistency.get("passed", False)
                )
            )

            def synthesize_current_answer() -> None:
                nonlocal answer, answer_review, context_engineering
                synthesis_started = time.perf_counter()
                answer, synthesis_trace = synthesize_answer_with_fallback(
                    query=retrieval_query,
                    plan=plan,
                    evidence_items=evidence_items,
                    min_evidence=args.min_evidence,
                    min_evidence_score=args.min_evidence_score,
                    max_claims=args.max_answer_claims,
                    llm_config=synthesis_config,
                    core_top_k=args.core_evidence_top_k,
                    support_top_k=args.support_evidence_top_k,
                    fallback_synthesizer=synthesize_grounded_answer,
                    max_context_tokens=args.llm_context_max_tokens,
                    max_tokens_per_evidence=args.llm_context_max_tokens_per_evidence,
                    context_dedup_threshold=args.llm_context_dedup_threshold,
                )
                context_engineering = dict(synthesis_trace.get("context_stats") or {})
                llm_calls.append(synthesis_trace)
                trace_recorder.add_llm_call(synthesis_trace)
                add_timing_value(timings, "synthesis_ms", round((time.perf_counter() - synthesis_started) * 1000, 2))
                if answer and getattr(args, "enable_answer_review", False):
                    answer_review_started = time.perf_counter()
                    answer, answer_review = review_answer_with_fallback(
                        query=retrieval_query,
                        plan=plan,
                        answer=answer,
                        evidence_items=evidence_items,
                        llm_config=answer_review_config,
                        core_top_k=args.core_evidence_top_k,
                        support_top_k=args.support_evidence_top_k,
                        max_context_tokens=args.llm_context_max_tokens,
                        max_tokens_per_evidence=args.llm_context_max_tokens_per_evidence,
                        context_dedup_threshold=args.llm_context_dedup_threshold,
                    )
                    llm_calls.append(answer_review)
                    trace_recorder.add_llm_call(answer_review)
                    add_timing_value(
                        timings,
                        "answer_review_ms",
                        round((time.perf_counter() - answer_review_started) * 1000, 2),
                    )

            if args.answer_mode == "grounded" and not synthesis_deferred:
                synthesize_current_answer()

            answer_for_reflection = (
                answer.to_dict()
                if answer
                else {
                    "status": "answered" if evidence_items and coverage.get("passed", False) and topic_consistency.get("passed", False) else "",
                    "refusal_reason": "",
                    "review_status": "",
                    "review_issues": [],
                }
            )
            reflection = {
                "sufficient": True,
                "missing_aspects": [],
                "rewritten_query": "",
                "reason": "reflection_disabled",
                "source": "disabled",
                "should_continue": False,
                "evidence_overlap_ratio": 0.0,
            }
            if getattr(args, "enable_reflection", False):
                reflection = reflect_on_evidence(
                    original_query=retrieval_query,
                    plan=plan.to_dict(),
                    evidence_items=evidence_payload,
                    answer=answer_for_reflection,
                    coverage=coverage,
                    topic_consistency=topic_consistency,
                    previous_queries=previous_queries,
                    hop_index=hop_index,
                    max_hops=max_hops,
                    llm_config=reflection_llm_config,
                )
                if reflection.get("llm_call"):
                    llm_calls.append(reflection["llm_call"])
                    trace_recorder.add_llm_call(reflection["llm_call"])

                overlap = max(
                    [evidence_overlap_ratio(previous, current_evidence_uids) for previous in previous_evidence_sets] or [0.0]
                )
                reflection["evidence_overlap_ratio"] = round(float(overlap), 4)

                rewritten_query = str(reflection.get("rewritten_query") or "").strip()
                normalized_previous_queries = {normalize_text(item) for item in previous_queries if normalize_text(item)}
                stop_reason = ""
                if reflection.get("sufficient"):
                    stop_reason = "reflection_says_sufficient"
                elif hop_index >= max_hops:
                    stop_reason = "max_hops_reached"
                elif not rewritten_query:
                    stop_reason = "empty_rewritten_query"
                elif normalize_text(rewritten_query) in normalized_previous_queries:
                    stop_reason = "duplicate_query"
                elif overlap >= float(getattr(args, "reflection_overlap_threshold", DEFAULT_REFLECTION_OVERLAP_THRESHOLD)):
                    stop_reason = "evidence_overlap_threshold"
                else:
                    reflection["should_continue"] = True
                    current_query = rewritten_query
                    previous_queries.append(rewritten_query)
                reflection["stop_reason"] = stop_reason
            reflections.append(reflection)

            if synthesis_deferred and not reflection.get("should_continue"):
                synthesize_current_answer()

            trace_recorder.add_hop(
                hop_index=hop_index,
                query=hop_query,
                retrieved=[item.to_dict() | {"score_breakdown": item.score_breakdown} for item in hop_output["candidates"]],
                reranked=[item.to_dict() | {"score_breakdown": item.score_breakdown} for item in reranked],
                evidence=evidence_payload,
                metadata={
                    "query_variants": list(hop_output["query_variants"]),
                    "retrieval_mode": args.retrieval_mode,
                    "parallel_hierarchical": bool(args.parallel_hierarchical),
                    "effective_sparse_retrieval": effective_sparse_retrieval,
                    "effective_bge_dense_retrieval": effective_bge_dense_retrieval,
                    "effective_bge_sparse_retrieval": effective_bge_sparse_retrieval,
                    "coverage": coverage,
                    "topic_consistency": topic_consistency,
                    "answer_review": answer_review,
                    "reflection": reflection,
                    "synthesis_deferred": synthesis_deferred,
                    "candidate_pool_size": len(candidate_pool),
                    "context_engineering": context_engineering,
                },
                top_k=getattr(args, "trace_top_k", DEFAULT_TRACE_TOP_K),
            )
            previous_evidence_sets.append(current_evidence_uids)

            if not reflection.get("should_continue"):
                break
    finally:
        if qwen_embedder is not None and args.no_embedder_cache:
            try:
                qwen_embedder.close()
            except Exception:
                pass
        if bge_helper is not None and args.no_embedder_cache:
            try:
                bge_helper.close()
            except Exception:
                pass
        client.close()

    timings["total_ms"] = round((time.perf_counter() - total_started) * 1000, 2)
    trace_recorder.set_evidence([item.to_dict() for item in evidence_items])
    trace_recorder.set_answer(answer.to_dict() if answer else {})
    trace_recorder.set_timings(timings)
    trace_recorder.set_metadata(
        {
            "intent": intent,
            "retrieval_mode": args.retrieval_mode,
            "parallel_hierarchical": bool(args.parallel_hierarchical),
            "effective_sparse_retrieval": effective_sparse_retrieval,
            "effective_bge_dense_retrieval": effective_bge_dense_retrieval,
            "effective_bge_sparse_retrieval": effective_bge_sparse_retrieval,
            "query_variants": accumulated_query_variants or query_variants,
            "instruction_queries": accumulated_instruction_queries or instruction_queries,
            "notices": notices,
            "answer_review": answer_review,
            "reflections": reflections,
            "context_engineering": context_engineering,
        }
    )
    trace_file = trace_recorder.write()

    if session_id and getattr(args, "enable_memory", False):
        turn = Turn(
            turn_id=turn_id,
            user_query=original_query,
            standalone_query=retrieval_query,
            intent=intent,
            task_type=plan.task_type,
            answer_status=str(answer.status if answer else ""),
            answer=str(answer.answer if answer else ""),
            summary=build_turn_summary(original_query, answer.answer if answer else ""),
            retrieved_chunk_uids=[item.chunk_uid for item in results],
            evidence_chunk_uids=[item.chunk_uid for item in evidence_items],
            plan=plan.to_dict(),
            timings=timings,
            trace_file=trace_file,
            metadata={
                "grounding_mode": answer.grounding_mode if answer else "",
                "llm_calls": llm_calls,
                "answer_review": answer_review,
                "reflections": reflections,
            },
        )
        memory.append_turn(
            session_id,
            turn,
            llm_config=contextualization_llm_config,
            llm_trace_collector=llm_calls,
        )

    return {
        "query": original_query,
        "standalone_query": retrieval_query,
        "session_id": session_id,
        "turn_id": turn_id,
        "intent": intent,
        "plan": plan.to_dict(),
        "retrieval_mode": args.retrieval_mode,
        "parallel_hierarchical": bool(args.parallel_hierarchical),
        "timings": timings,
        "query_variants": accumulated_query_variants or query_variants,
        "instruction_queries": accumulated_instruction_queries or instruction_queries,
        "contextualization": contextualization,
        "notices": notices,
        "answer_review": answer_review,
        "reflections": reflections,
        "context_engineering": context_engineering,
        "hop_count": len(reflections) or 1,
        "api_rerank_enabled": bool(args.enable_api_rerank),
        "rerank_provider": args.rerank_provider if args.enable_api_rerank else "",
        "effective_sparse_retrieval": effective_sparse_retrieval,
        "effective_bge_dense_retrieval": effective_bge_dense_retrieval,
        "effective_bge_sparse_retrieval": effective_bge_sparse_retrieval,
        "results": [item.to_dict() | {"score_breakdown": item.score_breakdown} for item in results],
        "evidence": [item.to_dict() for item in evidence_items],
        "answer": answer.to_dict() if answer else None,
        "trace_file": trace_file,
        "llm_calls": llm_calls,
    }


def _shorten_evidence_text(text: str, max_chars: int = 260) -> str:
    cleaned = cleanup_retrieval_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(20, max_chars - 3)].rstrip() + "..."


def _citation_source_label(item: Dict[str, Any]) -> str:
    doc_title = str(item.get("doc_title") or item.get("citation", {}).get("doc_title") or "").strip()
    section_title = str(item.get("section_title") or item.get("citation", {}).get("section_title") or "").strip()
    if doc_title and section_title:
        return f"{doc_title} / {section_title}"
    return doc_title or section_title or str(item.get("source_file") or item.get("citation", {}).get("source_file") or "").strip()


def _collect_llm_evidence(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    for call in output.get("llm_calls", []) or []:
        if call.get("type") != "grounded_synthesis":
            continue
        request = call.get("request") or {}
        evidence = request.get("evidence") or []
        if evidence:
            return [dict(item) for item in evidence if isinstance(item, dict)]
    return []


def _collect_display_evidence(output: Dict[str, Any], max_items: int) -> List[Dict[str, Any]]:
    llm_evidence = _collect_llm_evidence(output)
    answer_payload = output.get("answer") or {}
    answer_text = str(answer_payload.get("answer") or "")
    cited_ids = []
    for item in list(answer_payload.get("citations") or []) + re.findall(r"\[(E\d+)\]", answer_text):
        if isinstance(item, str) and item not in cited_ids:
            cited_ids.append(item)

    if llm_evidence:
        by_id = {str(item.get("id") or "").strip(): item for item in llm_evidence}
        selected = [by_id[evidence_id] for evidence_id in cited_ids if evidence_id in by_id]
        for item in llm_evidence:
            if item not in selected:
                selected.append(item)
            if len(selected) >= max(1, max_items):
                break
        return selected[: max(1, max_items)]

    selected = []
    for index, item in enumerate(output.get("evidence", []) or [], start=1):
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        copied.setdefault("id", f"E{index}")
        selected.append(copied)
        if len(selected) >= max(1, max_items):
            break
    return selected


def format_answer_only_output(output: Dict[str, Any], *, show_evidence: bool, evidence_top_k: int) -> str:
    answer_payload = output.get("answer") or {}
    answer_text = str(answer_payload.get("answer") or "").strip()
    if not answer_text:
        answer_text = str(answer_payload.get("refusal_reason") or "当前没有生成可用答案。").strip()

    if not show_evidence:
        return answer_text

    evidence_items = _collect_display_evidence(output, max_items=evidence_top_k)
    if not evidence_items:
        return answer_text

    lines = [answer_text, "", "本次使用的本地证据："]
    for index, item in enumerate(evidence_items, start=1):
        evidence_id = str(item.get("id") or f"E{index}").strip()
        label = _citation_source_label(item)
        source_file = str(item.get("source_file") or item.get("citation", {}).get("source_file") or "").strip()
        chunk_uid = str(item.get("chunk_uid") or item.get("citation", {}).get("chunk_uid") or "").strip()
        quote = _shorten_evidence_text(str(item.get("quote") or ""), max_chars=280)
        score = item.get("evidence_score")
        score_text = ""
        try:
            score_text = f"；证据分={float(score):.3f}" if score is not None else ""
        except (TypeError, ValueError):
            score_text = ""
        lines.append(f"{index}. [{evidence_id}] {label}{score_text}")
        if source_file:
            lines.append(f"   文件：{source_file}")
        if chunk_uid:
            lines.append(f"   chunk_uid：{chunk_uid}")
        if quote:
            lines.append(f"   摘录：{quote}")
    return "\n".join(lines)


def main() -> int:
    args = build_arg_parser().parse_args()
    preflight_runtime_checks(
        args.model_path,
        args.device,
        bge_m3_model_path=args.bge_m3_model_path,
        require_bge_m3=bool(
            (args.enable_bge_dense_retrieval or args.enable_bge_sparse_retrieval)
            and str(args.bge_m3_model_path or "").strip()
        ),
    )

    if isinstance(args.query_text, list):
        query = " ".join(str(part).strip() for part in args.query_text if str(part).strip()).strip()
    elif isinstance(args.query, list):
        query = " ".join(str(part).strip() for part in args.query if str(part).strip()).strip()
    else:
        query = str(args.query or "").strip()
    if not query:
        try:
            if not sys.stdin.isatty():
                query = sys.stdin.read().strip()
        except Exception:
            query = ""
    if not query:
        try:
            query = input("请输入问题: ").strip()
        except EOFError as exc:
            raise RuntimeError("No query provided and interactive input is unavailable.") from exc
    if not query:
        raise RuntimeError("Query cannot be empty.")

    output = run_search(args, query)

    if args.answer_only:
        print(
            format_answer_only_output(
                output,
                show_evidence=bool(args.show_evidence),
                evidence_top_k=max(1, int(args.answer_evidence_top_k)),
            )
        )
    elif args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_results(
            query=output["query"],
            standalone_query=output.get("standalone_query", output["query"]),
            intent=output["intent"],
            query_variants=output["query_variants"],
            results=[hydrate_search_candidate(item) for item in output["results"]],
            plan=hydrate_query_plan(output.get("plan") or {}, output["query"], output["intent"]),
            evidence_items=[hydrate_evidence_item(item) for item in output.get("evidence", [])],
            answer=hydrate_answer(output.get("answer")),
            timings=output.get("timings", {}),
            trace_file=output.get("trace_file", ""),
            notices=output.get("notices", []),
            verbose=args.verbose,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
