from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence, Tuple

from .models import EvidenceItem, QueryPlan


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_TERM_RE = re.compile(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}")
_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?%?|20\d{2}(?:[-/.]\d{1,2}){0,2}")
_INDUSTRY_RESEARCH_RE = re.compile(
    r"(?:市场规模|市场空间|增速|渗透率|供给|需求|产能|竞争格局|份额|集中度|"
    r"产业链|价值链|商业模式|毛利率|利润率|成本|价格|盈利|政策|风险|催化|驱动|"
    r"测算|口径|数据来源|报告|截至)"
)


def strip_retrieval_metadata(text: str) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"(?i)\bsource_ref:\s*chunk\s*=\s*\d+\s*span\s*=\s*chunk\s*\d+\s*header_path:[^；;。]*[；;]?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\bsection:\s*[^；;。]*[；;]?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\bcontent type:\s*[a-z_]+\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\bnumbers:[^；;。]*\blogic_text:\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\blogic_text:\s*", "", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" |;；")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def normalize_for_dedup(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def estimate_tokens(text: str) -> int:
    text = str(text or "")
    if not text:
        return 0
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_count = max(0, len(text) - cjk_count)
    return max(1, int(math.ceil(cjk_count * 0.9 + other_count / 3.8)))


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    text = str(text or "").strip()
    if estimate_tokens(text) <= max_tokens:
        return text
    if max_tokens <= 0:
        return ""
    # Conservative char budget for Chinese-heavy enterprise docs.
    max_chars = max(80, int(max_tokens * 1.35))
    clipped = text[:max_chars].rstrip()
    return clipped + ("..." if len(clipped) < len(text) else "")


def split_sentences(text: str) -> List[str]:
    parts = []
    for part in _SENTENCE_SPLIT_RE.split(str(text or "")):
        cleaned = part.strip()
        if cleaned:
            parts.append(cleaned)
    return parts or [str(text or "").strip()]


def plan_terms(query: str, plan: QueryPlan) -> List[str]:
    values: List[str] = []
    for source in [
        query,
        plan.normalized_query,
        " ".join(plan.entity_terms),
        " ".join(plan.theme_terms),
        " ".join(plan.constraint_terms),
        " ".join(plan.time_terms),
        " ".join(plan.evidence_focus),
    ]:
        for term in _TERM_RE.findall(str(source or "")):
            cleaned = normalize_text(term)
            if len(cleaned) >= 2 and cleaned not in values:
                values.append(cleaned)
    return values[:40]


def score_sentence(sentence: str, query_terms: Sequence[str], plan: QueryPlan, evidence_score: float) -> float:
    lowered = normalize_text(sentence)
    score = float(evidence_score) * 0.15
    score += 1.6 * sum(1 for term in query_terms if term and term in lowered)
    if plan.task_type in {"fact", "market", "comparison", "trend"} and _NUMERIC_RE.search(sentence):
        score += 0.8
    if plan.task_type in {"market", "trend", "root_cause", "status", "comparison"} and _INDUSTRY_RESEARCH_RE.search(sentence):
        score += 0.55
    if _NUMERIC_RE.search(sentence) and re.search(r"(?:口径|数据|报告|统计|测算|截至|同比|环比)", sentence):
        score += 0.25
    if 18 <= len(sentence) <= 180:
        score += 0.35
    if len(sentence) > 320:
        score -= 0.25
    return score


def compress_quote(
    *,
    quote: str,
    query_terms: Sequence[str],
    plan: QueryPlan,
    evidence_score: float,
    max_tokens: int,
) -> Tuple[str, bool]:
    quote = strip_retrieval_metadata(quote)
    if estimate_tokens(quote) <= max_tokens:
        return quote, False

    sentences = [item for item in split_sentences(quote) if item]
    scored = [
        (score_sentence(sentence, query_terms, plan, evidence_score), index, sentence)
        for index, sentence in enumerate(sentences)
    ]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    selected_indices = []
    used_tokens = 0
    for _, index, sentence in scored:
        sentence_tokens = estimate_tokens(sentence)
        if sentence_tokens > max_tokens:
            sentence = truncate_to_token_budget(sentence, max_tokens)
            sentence_tokens = estimate_tokens(sentence)
        if used_tokens + sentence_tokens > max_tokens and selected_indices:
            continue
        selected_indices.append(index)
        used_tokens += sentence_tokens
        if used_tokens >= max_tokens * 0.9:
            break

    if not selected_indices:
        return truncate_to_token_budget(quote, max_tokens), True

    selected = " ".join(sentences[index] for index in sorted(set(selected_indices))).strip()
    return truncate_to_token_budget(selected, max_tokens), True


def is_duplicate_quote(quote: str, seen_quotes: Sequence[str], threshold: float) -> bool:
    current = normalize_for_dedup(quote)
    if not current:
        return True
    for seen in seen_quotes:
        if not seen:
            continue
        if current == seen:
            return True
        if len(current) >= 80 and (current in seen or seen in current):
            return True
        if SequenceMatcher(None, current[:1200], seen[:1200]).ratio() >= threshold:
            return True
    return False


def rank_evidence_for_context(
    evidence_items: Sequence[EvidenceItem],
    *,
    core_top_k: int,
    support_top_k: int,
) -> List[EvidenceItem]:
    core_limit = max(1, int(core_top_k))
    support_limit = max(core_limit, int(support_top_k))
    ranked = sorted(
        evidence_items,
        key=lambda item: (
            str(item.tier or "") == "core",
            float(item.evidence_score),
            float(item.final_score),
            -int(item.rank or 0),
        ),
        reverse=True,
    )
    core = [item for item in ranked if str(item.tier or "") == "core"][:core_limit]
    support = [item for item in ranked if item not in core][: max(0, support_limit - len(core))]
    return core + support


def build_context_pack(
    *,
    query: str,
    plan: QueryPlan,
    evidence_items: Sequence[EvidenceItem],
    core_top_k: int,
    support_top_k: int,
    max_context_tokens: int,
    max_tokens_per_evidence: int,
    dedup_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, EvidenceItem], Dict[str, Any]]:
    max_context_tokens = max(400, int(max_context_tokens))
    max_tokens_per_evidence = max(80, int(max_tokens_per_evidence))
    dedup_threshold = max(0.50, min(0.99, float(dedup_threshold)))
    query_terms = plan_terms(query, plan)
    candidates = rank_evidence_for_context(
        evidence_items,
        core_top_k=core_top_k,
        support_top_k=support_top_k,
    )

    payload: List[Dict[str, Any]] = []
    index_map: Dict[str, EvidenceItem] = {}
    seen_quotes: List[str] = []
    used_tokens = 0
    dropped_duplicates = 0
    dropped_budget = 0
    compressed_count = 0

    for item in candidates:
        if used_tokens >= max_context_tokens:
            dropped_budget += 1
            continue
        remaining = max_context_tokens - used_tokens
        per_item_budget = min(max_tokens_per_evidence, max(80, remaining))
        quote, compressed = compress_quote(
            quote=item.quote,
            query_terms=query_terms,
            plan=plan,
            evidence_score=float(item.evidence_score),
            max_tokens=per_item_budget,
        )
        if not quote:
            continue
        if is_duplicate_quote(quote, seen_quotes, threshold=dedup_threshold):
            dropped_duplicates += 1
            continue

        quote_tokens = estimate_tokens(quote)
        if used_tokens + quote_tokens > max_context_tokens and payload:
            dropped_budget += 1
            continue

        evidence_id = f"E{len(payload) + 1}"
        index_map[evidence_id] = item
        seen_quotes.append(normalize_for_dedup(quote))
        used_tokens += quote_tokens
        if compressed:
            compressed_count += 1
        payload.append(
            {
                "id": evidence_id,
                "tier": "core" if len(payload) < max(1, int(core_top_k)) else "support",
                "rank": item.rank,
                "doc_title": item.doc_title,
                "section_title": item.section_title,
                "source_file": item.source_file,
                "chunk_uid": item.chunk_uid,
                "chunk_level": item.chunk_level,
                "group": item.group,
                "quote": quote,
                "evidence_score": round(float(item.evidence_score), 4),
                "final_score": round(float(item.final_score), 4),
                "citation": item.citation,
                "estimated_tokens": quote_tokens,
                "compressed": compressed,
            }
        )

    stats = {
        "input_evidence_count": len(evidence_items),
        "candidate_evidence_count": len(candidates),
        "packed_evidence_count": len(payload),
        "estimated_context_tokens": used_tokens,
        "max_context_tokens": max_context_tokens,
        "max_tokens_per_evidence": max_tokens_per_evidence,
        "compressed_count": compressed_count,
        "dropped_duplicate_count": dropped_duplicates,
        "dropped_budget_count": dropped_budget,
        "dedup_threshold": dedup_threshold,
    }
    return payload, index_map, stats
