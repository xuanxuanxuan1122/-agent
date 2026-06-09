from __future__ import annotations

import os
import re
from collections import Counter
from urllib.parse import urlparse
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rag_pipeline.contracts.source_registry import renumber_sources_by_first_citation as _contract_renumber_sources_by_first_citation
from .citation_manifest import (
    attach_manifest_citations,
    build_citation_manifest,
    evidence_source_entries_from_package,
    manifest_appendix_sources,
    merge_source_registries,
)
from .chapter_narrative_agent import run_chapter_narrative

from .markdown_renderer import (
    _key_data_bullet_from_table_row,
    collect_format_warnings,
    normalize_markdown_spacing,
    render_appendix,
    render_chapter_package,
    render_cover,
    render_decision_package,
    render_executive_summary,
    render_risk_package,
    render_table_package,
    strip_body_qa_leaks,
    strip_internal_layout_language,
)
from .public_report_sanitizer import (
    apply_public_narrative_gate,
    has_internal_gap_language,
    public_narrative_leak_audit,
    public_text_artifact_counts,
    rewrite_internal_gap_language,
    sanitize_public_markdown,
)
from .report_contracts import (
    filter_resolvable_evidence_refs,
    resolve_section_source_refs,
    text_has_factual_claim,
)


AGENT_NAME = "final_writer_agent"
AGENT_DESCRIPTION = "Final Writer Agent. Only composes structured packages and renders Markdown."
PUBLIC_SECTION_KEYS = {
    "section_id",
    "section_title",
    "block_type",
    "output_type",
    "section_role",
    "claim_id",
    "claim_strength",
    "analysis_role",
    "source_support_map",
    "paragraph_seed",
    "required_evidence_refs",
    "claim",
    "reasoning",
    "mechanism",
    "counter_evidence",
    "actionable",
    "decision_implication",
    "what_to_verify_next",
    "confidence",
    "evidence_refs",
    "used_fact_refs",
    "citation_refs",
    "supporting_facts",
    "render_blocks",
    "public_render",
    "layout_generated",
    "evidence_backed",
    "observation_only",
    "force_render_observation",
    "layout_match_score",
    "layout_match_reason",
}
CITATION_RE = re.compile(r"\[\d{1,5}\]")


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100_000) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))

SUMMARY_BLOCKS = {
    "executive_summary",
    "key_judgments",
    "deal_snapshot",
    "investment_conclusion",
    "policy_summary",
    "impact_judgment",
    "entry_decision_snapshot",
    "market_snapshot",
    "competitive_snapshot",
    "consumer_opportunity_snapshot",
    "supply_chain_snapshot",
    "technology_readiness_snapshot",
    "briefing_summary",
}

DECISION_BLOCKS = {
    "strategic_options",
    "entry_recommendation",
    "investment_recommendation",
    "product_opportunity",
    "resilience_options",
    "adoption_path",
}

RISK_BLOCKS = {"risk_triggers", "red_flags", "execution_risks"}
WATCHLIST_BLOCKS = {"verification_checklist", "monitoring_indicators", "dd_checklist"}

GLOBAL_BLOCK_TITLES = {
    "executive_summary": "核心观点与主要结论",
    "key_judgments": "关键判断",
    "key_data": "关键数据",
    "deal_snapshot": "交易速览",
    "investment_conclusion": "投资结论",
    "policy_summary": "政策摘要",
    "impact_judgment": "影响判断",
    "entry_decision_snapshot": "进入决策速览",
    "market_snapshot": "市场速览",
    "competitive_snapshot": "竞争速览",
    "consumer_opportunity_snapshot": "消费机会速览",
    "supply_chain_snapshot": "供应链速览",
    "technology_readiness_snapshot": "技术成熟度速览",
    "briefing_summary": "简报摘要",
    "strategic_options": "策略选择",
    "entry_recommendation": "进入建议",
    "investment_recommendation": "投资建议",
    "product_opportunity": "产品机会",
    "resilience_options": "韧性建设选项",
    "adoption_path": "落地路径",
    "risk_triggers": "风险提示与反向信号",
    "red_flags": "尽调红旗",
    "execution_risks": "执行风险",
    "verification_checklist": "验证清单",
    "monitoring_indicators": "监测指标",
    "dd_checklist": "尽调清单",
    "appendix": "来源附录",
}

PUBLIC_SUMMARY_BLOCKS = {
    "executive_summary",
    "key_judgments",
    "deal_snapshot",
    "investment_conclusion",
    "impact_judgment",
    "entry_decision_snapshot",
    "market_snapshot",
    "competitive_snapshot",
    "consumer_opportunity_snapshot",
    "supply_chain_snapshot",
    "technology_readiness_snapshot",
    "briefing_summary",
}
PUBLIC_POLICY_SUMMARY_PROFILES = {"policy_impact_report", "briefing_note"}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _is_deep_report_family(report_blueprint: Optional[Dict[str, Any]]) -> bool:
    family = str(_as_dict(report_blueprint).get("report_family") or _as_dict(report_blueprint).get("report_type") or "").strip().lower()
    return family == "industry_deep_report" or "deep" in family


def _source_appendix_enabled(report_blueprint: Optional[Dict[str, Any]] = None) -> bool:
    return _env_flag("REPORT_FINAL_WRITER_SOURCE_APPENDIX", _is_deep_report_family(report_blueprint))


def _public_policy_summary_allowed(report_blueprint: Optional[Dict[str, Any]]) -> bool:
    if _env_flag("REPORT_PUBLIC_POLICY_SUMMARY", False):
        return True
    payload = _as_dict(report_blueprint)
    profile_name = str(payload.get("name") or payload.get("profile_name") or payload.get("report_family") or payload.get("report_type") or "").strip()
    return profile_name in PUBLIC_POLICY_SUMMARY_PROFILES


def _public_global_block_allowed(block_key: str, report_blueprint: Optional[Dict[str, Any]]) -> bool:
    key = str(block_key or "").strip()
    if not key:
        return False
    if _env_flag("REPORT_PUBLIC_RENDER_DIAGNOSTIC_BLOCKS", False):
        return True
    if key == "appendix":
        return True
    if key == "policy_summary":
        return _public_policy_summary_allowed(report_blueprint)
    if key in PUBLIC_SUMMARY_BLOCKS:
        return True
    return False


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _dedupe_strings(values: Sequence[Any], *, limit: int = 200) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _public_text(value: Any) -> str:
    text = rewrite_internal_gap_language(str(value or "").strip())
    return "" if has_internal_gap_language(text) else text


def _source_host(source: Dict[str, Any]) -> str:
    url = str(source.get("url") or source.get("source_url") or "").strip()
    if not url:
        return ""
    parsed = urlparse(url if re.match(r"^[a-z][a-z0-9+.-]*://", url, flags=re.I) else f"https://{url}")
    return str(parsed.netloc or parsed.path.split("/")[0] or "").strip().lower()


def _stable_local_source(source: Dict[str, Any]) -> bool:
    doc_id = str(source.get("document_id") or source.get("doc_id") or source.get("page_ref") or "").strip()
    title = str(source.get("title") or source.get("source_title") or "").strip()
    publisher = str(source.get("publisher") or source.get("source") or "").strip()
    date = str(source.get("date") or source.get("published_at") or "").strip()
    return bool(doc_id and sum(bool(item) for item in (title, publisher, date)) >= 2)


def _source_is_traceable(source: Dict[str, Any]) -> bool:
    return bool(str(source.get("url") or source.get("source_url") or "").strip() or _stable_local_source(source))


_GENERIC_REPORT_SOURCE_TITLE_RE = re.compile(
    r"^(?:"
    r"official(?:\s+(?:ai\s+agent\s+)?(?:statistics|data|source|report|disclosure))?"
    r"|official\s+statistics\s+show"
    r"|[\w.-]+\s+source"
    r"|source"
    r")$",
    flags=re.I,
)


def _generic_report_source_title(title: Any) -> bool:
    text = re.sub(r"\s+", " ", str(title or "").strip())
    return bool(not text or _GENERIC_REPORT_SOURCE_TITLE_RE.fullmatch(text))


def _repair_generic_source(source: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(source)
    host = _source_host(copied)
    title = str(copied.get("title") or copied.get("source_title") or "").strip()
    if _generic_report_source_title(title) and host:
        copied.setdefault("publisher", host)
        copied.setdefault("source", host)
        copied["generic_title_needs_repair"] = True
        copied["title"] = f"{host} source"
        copied["source_title"] = f"{host} source"
    return copied


LOW_QUALITY_REPORT_SOURCE_DOMAINS = (
    "twitter.com",
    "x.com",
    "instagram.com",
    "facebook.com",
    "baike.baidu.com",
    "baijiahao.baidu.com",
    "blog.csdn.net",
    "cnblogs.com",
    "juejin.cn",
    "youtube.com",
    "youtu.be",
    "linkedin.com",
    "dfans.xyz",
    "gatexx.com",
    "fxbaogao.com",
    "sgpjbg.com",
    "jazzyear.com",
    "zbj.com",
)

LOW_QUALITY_REPORT_SOURCE_TITLE_PATTERNS = (
    r"\u519c\u4e1a\u4eba\u5de5\u667a\u80fd",
    r"\u6570\u636e\u6295\u6bd2",
    r"\u7eba\u7ec7",
    r"\u667a\u80fd\u624b\u673a",
    r"\u53d1\u73b0\u62a5\u544a",
    r"Scribd",
    r"mrdeepfakes",
    r"\u4ee5\u4e0b\u662f\u5bf9\u6574\u7bc7.*(?:\u6df1\u5ea6\u5206\u6790|\u6846\u67b6\u63d0\u70bc)",
    r"免费看",
    r"SEO",
    r"search/newsflashes",
)


_QUERY_TOPIC_NEGATIVE_TERMS = {
    "ai_agent": (
        ("物价走势", "化工市场", "中国石油", "跨境电商", "纺织", "智能手机"),
        ("ai", "agent", "智能体", "人工智能", "大模型", "aigc", "算力", "模型"),
    ),
}

_QUERY_TOPIC_KEYWORDS = {
    "ai_agent": ("ai agent", "agentic", "智能体", "ai智能体", "ai 智能体"),
}


def _detect_query_topic(query_text: str) -> str:
    """Map a free-text query to a known topic id used for source filtering.

    Topic-specific negative term lists live in `_QUERY_TOPIC_NEGATIVE_TERMS`
    keyed by the topic id. New domains (储能、光伏、机器人 …) should be added
    here rather than hardcoded inline. Returns an empty string when no
    topic matches — meaning no topic-specific filter will be applied.
    """

    text = str(query_text or "").lower()
    if not text:
        return ""
    for topic_id, keywords in _QUERY_TOPIC_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return topic_id
    return ""


def _topic_filter_excludes_source(topic_id: str, *, identity: str, identity_lower: str) -> bool:
    """Return True if `topic_id`'s blocklist excludes this source.

    The contract: a source is excluded only when one of the topic's
    `unrelated_terms` is present in the identity AND none of the
    `positive_terms` rescues it. This avoids both false positives (the
    blocklist firing on every other report) and false negatives (a
    petroleum article slipping into an AI Agent report).
    """

    config = _QUERY_TOPIC_NEGATIVE_TERMS.get(topic_id)
    if not config:
        return False
    unrelated_terms, positive_terms = config
    if not any(term in identity for term in unrelated_terms):
        return False
    return not any(term in identity_lower or term in identity for term in positive_terms)


def _source_allowed_for_report(source: Dict[str, Any], *, query: str = "") -> bool:
    host = _source_host(source).lower()
    title = str(source.get("title") or source.get("source_title") or "").strip()
    identity = " ".join(
        str(source.get(key) or "")
        for key in ("ref", "url", "source_url", "title", "source_title", "publisher", "source")
    )
    query_text = str(query or "").lower()
    identity_lower = identity.lower()
    source_type = str(source.get("source_type") or source.get("type") or "").strip().lower()
    source_level = str(source.get("source_level") or source.get("credibility") or "").strip().upper()
    if source_level == "D":
        return False
    if source.get("source_title_url_mismatch_suspected"):
        return False
    if re.search(r"\bIQS\s*来源\b|^IQS来源$|example\.(?:com|gov|org)", identity, flags=re.I):
        return False
    if source_type in {"self_media", "social", "forum", "wiki", "seo", "search_page", "aggregator"}:
        return False
    if any(re.search(pattern, title, flags=re.I) for pattern in LOW_QUALITY_REPORT_SOURCE_TITLE_PATTERNS):
        return False
    if any(host == domain or host.endswith("." + domain) for domain in LOW_QUALITY_REPORT_SOURCE_DOMAINS):
        return False
    topic_id = _detect_query_topic(query_text)
    if topic_id and _topic_filter_excludes_source(
        topic_id, identity=identity, identity_lower=identity_lower
    ):
        return False
    return True


def _traceable_source_registry(source_registry: Sequence[Dict[str, Any]], *, query: str = "") -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    title_hosts: Dict[str, set[str]] = {}
    for source in list(source_registry or []):
        if not isinstance(source, dict):
            continue
        title = re.sub(r"\s+", " ", str(source.get("title") or source.get("source_title") or "").strip()).lower()
        if _generic_report_source_title(title):
            continue
        host = _source_host(source)
        if title and host:
            title_hosts.setdefault(title, set()).add(host)
    for source in list(source_registry or []):
        if not isinstance(source, dict):
            continue
        repaired = _repair_generic_source(source)
        title_key = re.sub(r"\s+", " ", str(repaired.get("title") or repaired.get("source_title") or "").strip()).lower()
        host = _source_host(repaired)
        if title_key and host and len(title_hosts.get(title_key, set())) > 1:
            repaired["source_title_url_mismatch_suspected"] = True
            repaired["original_title"] = repaired.get("title") or repaired.get("source_title")
            repaired["title"] = f"{host} source"
        if _source_is_traceable(repaired) and _source_allowed_for_report(repaired, query=query):
            kept.append(repaired)
        else:
            repaired["report_exclusion_reason"] = _source_report_exclusion_reason(repaired, query=query)
            excluded.append(repaired)
    return kept, excluded


def _source_identity_refs(source: Dict[str, Any]) -> set[str]:
    refs = {
        str(source.get("ref") or "").strip(),
        str(source.get("evidence_id") or "").strip(),
        str(source.get("source_ref") or "").strip(),
        str(source.get("citation_ref") or "").strip(),
        str(source.get("document_id") or "").strip(),
        str(source.get("doc_id") or "").strip(),
    }
    for key in ("evidence_refs", "used_fact_refs", "source_refs"):
        refs.update(str(item or "").strip() for item in _as_list(source.get(key)))
    return {ref for ref in refs if ref}


def _source_ref_lookup(source_registry: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for index, source in enumerate(source_registry, start=1):
        if not isinstance(source, dict):
            continue
        public_ref = str(source.get("ref") or "").strip() or f"[{index}]"
        for key in _source_identity_refs(source):
            lookup.setdefault(key, public_ref)
    return lookup


def _map_evidence_refs_to_source_refs(refs: Sequence[Any], lookup: Dict[str, str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for ref in refs:
        text = str(ref or "").strip()
        if not text:
            continue
        mapped = lookup.get(text, text)
        for candidate in (text, mapped):
            if candidate and candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    return result


def _attach_claim_source_refs_to_chapters(
    chapter_packages: Sequence[Dict[str, Any]],
    source_registry: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    lookup = _source_ref_lookup(source_registry)
    if not lookup:
        return [dict(chapter) for chapter in chapter_packages]
    chapters: List[Dict[str, Any]] = []
    for chapter in chapter_packages:
        if not isinstance(chapter, dict):
            continue
        copied = dict(chapter)
        sections = []
        for section in _as_list(copied.get("sections")):
            if not isinstance(section, dict):
                continue
            section_copy = dict(section)
            refs = _as_list(section_copy.get("evidence_refs")) or _as_list(section_copy.get("used_fact_refs"))
            mapped_refs = _map_evidence_refs_to_source_refs(refs, lookup)
            if mapped_refs:
                section_copy["evidence_refs"] = mapped_refs
                section_copy["used_fact_refs"] = _map_evidence_refs_to_source_refs(
                    _as_list(section_copy.get("used_fact_refs")) or refs,
                    lookup,
                )
            sections.append(section_copy)
        copied["sections"] = sections
        chapters.append(copied)
    return chapters


def _filter_rendered_refs_with_source_registry(
    *,
    chapters: Sequence[Dict[str, Any]],
    claim_units: Sequence[Dict[str, Any]],
    source_registry: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "filtered_refs": [],
        "filtered_unresolved_ref_count": 0,
        "sections_with_filtered_refs": 0,
        "claims_with_filtered_refs": 0,
        "factual_section_without_resolved_ref_count": 0,
        "section_ref_recovered_count": 0,
        "recovered_ref_examples": [],
        "citationless_fact_examples": [],
    }

    def _filter_refs(refs: Sequence[Any], *, context: Dict[str, Any]) -> List[str]:
        result = filter_resolvable_evidence_refs(refs, source_registry)
        filtered = []
        for item in result.get("filtered_refs", []):
            payload = dict(item)
            payload.update(context)
            filtered.append(payload)
        if filtered:
            diagnostics["filtered_refs"].extend(filtered)
            diagnostics["filtered_unresolved_ref_count"] += len(filtered)
        return list(result.get("resolved_refs") or [])

    def _inline_citations(payload: Dict[str, Any]) -> List[str]:
        values: List[str] = []
        for key in ("claim", "reasoning", "mechanism", "counter_evidence"):
            values.append(str(payload.get(key) or ""))
        for block in _as_list(payload.get("render_blocks")):
            if isinstance(block, dict):
                values.append(str(block.get("text") or ""))
        refs: List[str] = []
        seen = set()
        for value in values:
            for match in CITATION_RE.finditer(value):
                ref = match.group(0)
                if ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
        return refs

    def _recover_refs_from_section_facts(payload: Dict[str, Any]) -> List[str]:
        raw_refs: List[Any] = []
        for key in ("supporting_facts", "evidence_basis", "fact_cards", "facts"):
            for item in _as_list(payload.get(key)):
                if not isinstance(item, dict):
                    continue
                for field in (
                    "evidence_id",
                    "ref",
                    "source_ref",
                    "citation_ref",
                    "document_id",
                    "doc_id",
                    "url",
                    "source_url",
                ):
                    value = item.get(field)
                    if value not in (None, ""):
                        raw_refs.append(value)
                for nested_key in ("source", "metadata"):
                    nested = item.get(nested_key)
                    if not isinstance(nested, dict):
                        continue
                    for field in ("ref", "source_ref", "evidence_id", "citation_ref", "url", "source_url"):
                        value = nested.get(field)
                        if value not in (None, ""):
                            raw_refs.append(value)
        if not raw_refs:
            return []
        return list(filter_resolvable_evidence_refs(raw_refs, source_registry).get("resolved_refs") or [])

    cleaned_chapters: List[Dict[str, Any]] = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        chapter_copy = dict(chapter)
        sections = []
        for section in _as_list(chapter_copy.get("sections")):
            if not isinstance(section, dict):
                continue
            section_copy = dict(section)
            inline_refs = _inline_citations(section_copy)
            if inline_refs and not any(
                _as_list(section_copy.get(key))
                for key in ("citation_refs", "evidence_refs", "used_fact_refs", "required_evidence_refs", "source_refs")
            ):
                section_copy["citation_refs"] = inline_refs
            recovered_refs = _recover_refs_from_section_facts(section_copy)
            if recovered_refs:
                existing_refs = _as_list(section_copy.get("used_fact_refs"))
                section_copy["used_fact_refs"] = _dedupe_strings([*existing_refs, *recovered_refs])
                if not _as_list(section_copy.get("evidence_refs")):
                    section_copy["evidence_refs"] = list(section_copy["used_fact_refs"])
                diagnostics["section_ref_recovered_count"] += 1
                if len(diagnostics["recovered_ref_examples"]) < 8:
                    diagnostics["recovered_ref_examples"].append(
                        {
                            "chapter_id": chapter_copy.get("chapter_id"),
                            "section_id": section_copy.get("section_id"),
                            "recovered_refs": list(recovered_refs)[:5],
                        }
                    )
            section_filtered = False
            for key in ("citation_refs", "evidence_refs", "used_fact_refs", "required_evidence_refs", "source_refs"):
                if key not in section_copy:
                    continue
                refs = _as_list(section_copy.get(key))
                resolved = _filter_refs(
                    refs,
                    context={
                        "chapter_id": chapter_copy.get("chapter_id"),
                        "section_id": section_copy.get("section_id"),
                        "field": key,
                    },
                )
                if len(resolved) != len([ref for ref in refs if str(ref or "").strip()]):
                    section_filtered = True
                section_copy[key] = resolved
            if section_filtered:
                section_copy["unresolved_refs_filtered"] = True
                diagnostics["sections_with_filtered_refs"] += 1
            section_lineage = resolve_section_source_refs(section_copy, source_registry)
            if section_lineage.get("has_factual_claim") and not section_lineage.get("has_resolved_ref"):
                section_copy["factual_claim_without_resolved_ref"] = True
                section_copy["source_ref_resolution_reason"] = section_lineage.get("reason")
                diagnostics["factual_section_without_resolved_ref_count"] += 1
                if len(diagnostics["citationless_fact_examples"]) < 8:
                    diagnostics["citationless_fact_examples"].append(
                        {
                            "chapter_id": chapter_copy.get("chapter_id"),
                            "section_id": section_copy.get("section_id"),
                            "claim": str(section_copy.get("claim") or section_copy.get("reasoning") or "")[:220],
                        }
                    )
            sections.append(section_copy)
        chapter_copy["sections"] = sections
        cleaned_chapters.append(chapter_copy)

    cleaned_claims: List[Dict[str, Any]] = []
    for claim in claim_units:
        if not isinstance(claim, dict):
            continue
        claim_copy = dict(claim)
        claim_filtered = False
        for key in ("evidence_refs", "used_fact_refs", "used_evidence_ids", "supporting_evidence_refs", "supporting_evidence"):
            if key not in claim_copy:
                continue
            refs = _as_list(claim_copy.get(key))
            resolved = _filter_refs(
                refs,
                context={
                    "chapter_id": claim_copy.get("chapter_id"),
                    "section_id": claim_copy.get("section_id"),
                    "field": key,
                    "claim_id": claim_copy.get("claim_id") or claim_copy.get("id"),
                },
            )
            if len(resolved) != len([ref for ref in refs if str(ref or "").strip()]):
                claim_filtered = True
            claim_copy[key] = resolved
        if claim_filtered:
            claim_copy["unresolved_refs_filtered"] = True
            diagnostics["claims_with_filtered_refs"] += 1
        cleaned_claims.append(claim_copy)

    return cleaned_chapters, cleaned_claims, diagnostics


def _renumber_public_chapter_headings(markdown: str) -> str:
    chapter_index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal chapter_index
        chapter_index += 1
        title = re.sub(r"^\d+\.\s*", "", match.group(1).strip())
        return f"## {chapter_index}. {title}"

    return re.sub(r"^##\s+\d+\.\s+(.+?)\s*$", replace, markdown or "", flags=re.M)


def _rewrite_bare_metric_lines(markdown: str) -> tuple[str, int]:
    metric_keywords = r"规模|出货|部署|渗透|份额|收入|利润|成本|价格|订单|采购|客户|占比|增速|融资|市场|金额|数量"
    rewrite_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal rewrite_count
        label = match.group("label").strip()
        value = match.group("value").strip().rstrip("。；;")
        if not re.search(metric_keywords, label):
            return match.group(0)
        rewrite_count += 1
        if re.match(r"^(?:达|达到|超|超过|约|近|突破)", value):
            return f"{label}{value}。"
        return f"{label}为{value}。"

    rewritten = re.sub(
        r"^(?P<label>[^:：\n]{2,40})[:：]\s*(?P<value>[^。；;\n]{1,80})[。；;]?$",
        replace,
        str(markdown or ""),
        flags=re.M,
    )
    return rewritten, rewrite_count


def _drop_residual_headline_lines(markdown: str) -> tuple[str, int]:
    cleaned_lines: List[str] = []
    dropped_count = 0
    for line in str(markdown or "").splitlines():
        text = line.strip()
        if re.search(r"赛道[^。]{0,20}(?:爆发|加速|火热|红利)", text):
            dropped_count += 1
            continue
        if re.match(r"^(?:构建|打造|推出|发布|上线).{4,60}$", text) and not re.search(r"[。；;]", text):
            dropped_count += 1
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines), dropped_count


def _public_section(section: Dict[str, Any]) -> Dict[str, Any]:
    copied = {key: section.get(key) for key in PUBLIC_SECTION_KEYS if key in section}
    for key in ["section_title", "claim", "reasoning", "mechanism", "counter_evidence", "actionable", "decision_implication", "confidence"]:
        if key in copied:
            copied[key] = _public_text(copied.get(key))
    copied["what_to_verify_next"] = [
        item
        for item in (_public_text(value) for value in _as_list(copied.get("what_to_verify_next")))
        if item
    ]
    copied_blocks = []
    for block in (_as_dict(item) for item in _as_list(copied.get("render_blocks"))):
        if not str(block.get("type") or "").strip():
            continue
        block = dict(block)
        block["label"] = _public_text(block.get("label"))
        block["text"] = _public_text(block.get("text"))
        copied_blocks.append(block)
    copied["render_blocks"] = copied_blocks
    return copied


def _section_has_public_content(section: Dict[str, Any]) -> bool:
    visible = " ".join(
        str(section.get(key) or "")
        for key in ["section_title", "claim", "reasoning", "mechanism", "counter_evidence", "actionable", "decision_implication"]
    )
    visible = " ".join([visible, *[str(_as_dict(block).get("text") or "") for block in _as_list(section.get("render_blocks"))]])
    visible = " ".join([visible, *[str(item) for item in _as_list(section.get("what_to_verify_next"))]])
    return bool(visible.strip()) and not has_internal_gap_language(visible)


def _all_table_packages(chapter_packages: Sequence[Dict[str, Any]], explicit: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = [item for item in explicit if isinstance(item, dict)]
    seen = {str(item.get("table_id") or id(item)) for item in result}
    for chapter in chapter_packages:
        if not isinstance(chapter, dict):
            continue
        for table in _as_list(chapter.get("table_packages")):
            if not isinstance(table, dict):
                continue
            key = str(table.get("table_id") or id(table))
            if key in seen:
                continue
            seen.add(key)
            result.append(table)
    return result


def _table_passed_for_public(table: Dict[str, Any]) -> bool:
    table = _as_dict(table)
    if not table.get("should_render") or table.get("appendix_only"):
        return False
    validation = _as_dict(table.get("validation") or table.get("table_validation_for_clean"))
    if validation and validation.get("passed") is False:
        return False
    if table.get("validation_error") or table.get("table_validation_error"):
        return False
    if _as_list(table.get("reject_reasons")):
        return False
    rows = _as_list(table.get("rows"))
    if len(rows) < 3:
        return False
    if str(table.get("metric_validation_status") or "").strip().lower() == "invalid":
        return False
    for row in rows:
        if str(_as_dict(row).get("metric_validation_status") or "").strip().lower() == "invalid":
            return False
    return True


def _renumber_sources_by_first_citation(
    markdown: str,
    source_registry: Sequence[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    return _contract_renumber_sources_by_first_citation(markdown, source_registry)


def _sources_cited_in_body(markdown: str, source_registry: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cited_refs = {f"[{match.group(1)}]" for match in re.finditer(r"\[(\d{1,5})\]", str(markdown or ""))}
    if not cited_refs:
        return []
    return [
        dict(source)
        for source in list(source_registry or [])
        if isinstance(source, dict) and _normalize_citation_ref(source.get("ref")) in cited_refs
    ]


def _normalize_citation_ref(value: Any) -> str:
    """Normalize a citation token to the canonical `[N]` form.

    Sources may store refs as ``"1"`` (raw number), ``"[1]"`` (with
    brackets), or even ``"[ 1 ]"`` (with stray whitespace). The downstream
    body markdown only ever contains ``[N]``, so we coerce everything to
    that form before comparing.
    """

    text = str(value or "").strip()
    if not text:
        return ""
    match = re.fullmatch(r"\[?\s*(\d{1,5})\s*\]?", text)
    if match:
        return f"[{match.group(1)}]"
    return text


def _strip_orphan_citations(markdown: str, source_registry: Sequence[Dict[str, Any]]) -> str:
    valid_refs = {
        _normalize_citation_ref(source.get("ref"))
        for source in list(source_registry or [])
        if isinstance(source, dict) and str(source.get("ref") or "").strip()
    }
    valid_refs.discard("")
    if not valid_refs:
        return re.sub(r"\[\d{1,5}\]", "", str(markdown or ""))

    def replace(match: re.Match[str]) -> str:
        ref = f"[{match.group(1)}]"
        return ref if ref in valid_refs else ""

    return re.sub(r"\[(\d{1,5})\]", replace, str(markdown or ""))


def _final_source_aliases(source: Dict[str, Any]) -> set[str]:
    aliases = {
        str(source.get("ref") or "").strip(),
        str(source.get("original_ref") or "").strip(),
        str(source.get("id") or "").strip(),
        str(source.get("evidence_id") or "").strip(),
        str(source.get("source_ref") or "").strip(),
        str(source.get("citation_ref") or "").strip(),
        str(source.get("document_id") or "").strip(),
        str(source.get("doc_id") or "").strip(),
    }
    for key in ("evidence_refs", "used_fact_refs", "source_refs", "refs"):
        aliases.update(str(item or "").strip() for item in _as_list(source.get(key)))
    normalized = {alias for alias in aliases if alias}
    normalized.update(_normalize_citation_ref(alias) for alias in list(normalized))
    normalized.discard("")
    return normalized


def _final_source_lookup(
    citation_manifest: Dict[str, Any],
    source_registry: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    manifest_sources = [
        item for item in _as_list(_as_dict(citation_manifest).get("appendix_sources")) if isinstance(item, dict)
    ]
    manifest_public_ref_owner: Dict[str, int] = {}
    for source in manifest_sources:
        public_ref = _normalize_citation_ref(source.get("ref"))
        if not public_ref:
            continue
        lookup[public_ref] = source
        manifest_public_ref_owner[public_ref] = id(source)
    sources = [
        *manifest_sources,
        *[item for item in list(source_registry or []) if isinstance(item, dict)],
    ]
    for source in sources:
        for alias in _final_source_aliases(source):
            normalized_alias = _normalize_citation_ref(alias)
            if (
                normalized_alias in manifest_public_ref_owner
                and manifest_public_ref_owner[normalized_alias] != id(source)
            ):
                continue
            lookup.setdefault(alias, source)
    for evidence_ref, citation_ref in _as_dict(_as_dict(citation_manifest).get("evidence_to_citation")).items():
        source = lookup.get(str(citation_ref or "").strip()) or lookup.get(_normalize_citation_ref(citation_ref))
        if source:
            lookup.setdefault(str(evidence_ref or "").strip(), source)
            lookup.setdefault(_normalize_citation_ref(evidence_ref), source)
    return {key: value for key, value in lookup.items() if key}


def _source_report_exclusion_reason(source: Dict[str, Any], *, query: str = "") -> str:
    blob = " ".join(
        str(source.get(key) or "")
        for key in ("title", "source_title", "summary", "snippet", "url", "source_url")
    ).lower()
    if re.search(r"404|not\s*found|页面未找到|页面不存在", blob, flags=re.I):
        return "dead_link"
    if source.get("source_title_url_mismatch_suspected"):
        return "source_mismatch"
    identity = " ".join(
        str(source.get(key) or "")
        for key in ("ref", "url", "source_url", "title", "source_title", "publisher", "source")
    )
    if re.search(r"\bIQS\s*来源\b|^IQS来源$|example\.(?:com|gov|org)", identity, flags=re.I):
        return "fake_or_placeholder_source"
    if not _source_is_traceable(source):
        return "untraceable_source"
    source_type = str(source.get("source_type") or source.get("type") or "").strip().lower()
    if source_type in {"self_media", "social", "forum", "wiki", "seo", "search_page", "aggregator"}:
        return "low_quality_source_type"
    title = str(source.get("title") or source.get("source_title") or "").strip()
    if any(re.search(pattern, title, flags=re.I) for pattern in LOW_QUALITY_REPORT_SOURCE_TITLE_PATTERNS):
        return "low_quality_title"
    host = _source_host(source)
    if any(host == domain or host.endswith("." + domain) for domain in LOW_QUALITY_REPORT_SOURCE_DOMAINS):
        return "low_quality_domain"
    topic_id = _detect_query_topic(str(query or "").lower())
    if topic_id and _topic_filter_excludes_source(topic_id, identity=identity, identity_lower=identity.lower()):
        return "topic_mismatch"
    source_level = str(source.get("source_level") or source.get("credibility") or "").strip().upper()
    if source_level == "D":
        return "source_level_d"
    return "not_allowed_for_report"


def _filtered_unresolved_ref_reason_details(
    refs: Sequence[Any],
    *,
    full_source_registry: Sequence[Dict[str, Any]],
    excluded_sources: Sequence[Dict[str, Any]],
    query: str = "",
) -> List[Dict[str, Any]]:
    if not refs:
        return []
    available_lookup = _final_source_lookup({}, full_source_registry)
    excluded_lookup = _final_source_lookup({}, excluded_sources)
    details: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        text = str(ref or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        source = excluded_lookup.get(text) or excluded_lookup.get(_normalize_citation_ref(text))
        if source:
            details.append(
                {
                    "ref": text,
                    "reason": source.get("report_exclusion_reason") or _source_report_exclusion_reason(source, query=query),
                    "title": source.get("title") or source.get("source_title"),
                    "url": source.get("url") or source.get("source_url"),
                }
            )
            continue
        if available_lookup.get(text) or available_lookup.get(_normalize_citation_ref(text)):
            details.append({"ref": text, "reason": "filtered_from_public_registry"})
        else:
            details.append({"ref": text, "reason": "missing_from_source_registry"})
    return details


def _body_citation_refs(markdown: str) -> List[str]:
    refs: List[str] = []
    seen = set()
    for match in re.finditer(r"\[(\d{1,5})\]", str(markdown or "")):
        ref = f"[{match.group(1)}]"
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    return refs


def _collapse_adjacent_duplicate_citations(markdown: str) -> tuple[str, int]:
    removed_count = 0

    def replace_sequence(match: re.Match[str]) -> str:
        nonlocal removed_count
        refs = [f"[{item}]" for item in re.findall(r"\[(\d{1,5})\]", match.group(0))]
        if len(refs) < 2:
            return match.group(0)
        collapsed: List[str] = []
        previous = ""
        for ref in refs:
            if ref == previous:
                removed_count += 1
                continue
            collapsed.append(ref)
            previous = ref
        return "".join(collapsed)

    rewritten = re.sub(r"(?:\[\d{1,5}\][ \t]*){2,}", replace_sequence, str(markdown or ""))
    return rewritten, removed_count


def _citationless_factual_segments(markdown: str, *, limit: int = 8) -> List[str]:
    examples: List[str] = []
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("|"):
            continue
        if _line_has_terminal_citation(line):
            continue
        if re.match(r"^\s*[-*]\s*$", line):
            continue
        for segment in _citationless_factual_sentence_examples(line):
            examples.append(segment[:260])
            if len(examples) >= limit:
                return examples
    return examples


def _drop_citationless_factual_bullets(markdown: str, *, limit: int = 12) -> tuple[str, Dict[str, Any]]:
    kept_lines: List[str] = []
    removed_examples: List[str] = []
    removed_count = 0
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.strip()
        bullet = bool(re.match(r"^\s*[-*]\s+\S", raw_line))
        if (
            bullet
            and line
            and not CITATION_RE.search(line)
            and text_has_factual_claim(line)
        ):
            removed_count += 1
            if len(removed_examples) < limit:
                removed_examples.append(line[:260])
            continue
        kept_lines.append(raw_line)
    return "\n".join(kept_lines), {
        "citationless_factual_bullet_removed_count": removed_count,
        "citationless_factual_bullet_removed_examples": removed_examples,
    }


def _drop_short_citationless_factual_lines(markdown: str, *, limit: int = 12) -> tuple[str, Dict[str, Any]]:
    kept_lines: List[str] = []
    removed_examples: List[str] = []
    removed_count = 0
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("|"):
            kept_lines.append(raw_line)
            continue
        if CITATION_RE.search(line) or re.match(r"^\s*[-*]\s+\S", raw_line):
            kept_lines.append(raw_line)
            continue
        if len(line) >= 120 or not text_has_factual_claim(line):
            kept_lines.append(raw_line)
            continue
        removed_count += 1
        if len(removed_examples) < limit:
            removed_examples.append(line[:260])
    return "\n".join(kept_lines), {
        "citationless_short_factual_line_removed_count": removed_count,
        "citationless_short_factual_line_removed_examples": removed_examples,
    }


_SENTENCE_SPLIT_RE = re.compile(r"[^。！？!?.；;\n]+[。！？!?.；;]?")


def _line_has_terminal_citation(line: str) -> bool:
    return bool(re.search(r"(?:\[\d{1,5}\]\s*)+$", str(line or "").strip()))


def _sentence_units(line: str) -> List[str]:
    units = [match.group(0).strip() for match in _SENTENCE_SPLIT_RE.finditer(str(line or "")) if match.group(0).strip()]
    return units or ([str(line or "").strip()] if str(line or "").strip() else [])


def _citationless_factual_sentence_examples(line: str) -> List[str]:
    text = str(line or "").strip()
    if not text or not text_has_factual_claim(text):
        return []
    if _line_has_terminal_citation(text):
        return []
    if not CITATION_RE.search(text):
        return [text] if text_has_factual_claim(text) else []
    examples: List[str] = []
    for unit in _sentence_units(text):
        if CITATION_RE.search(unit):
            continue
        if text_has_factual_claim(unit):
            examples.append(unit)
    return examples


def _drop_citationless_factual_sentences(markdown: str, *, limit: int = 12) -> tuple[str, Dict[str, Any]]:
    kept_lines: List[str] = []
    removed_examples: List[str] = []
    removed_count = 0
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("|") or re.match(r"^\s*[-*]\s+\S", raw_line):
            kept_lines.append(raw_line)
            continue
        if not text_has_factual_claim(line) or _line_has_terminal_citation(line):
            kept_lines.append(raw_line)
            continue
        if not CITATION_RE.search(line):
            removed_count += 1
            if len(removed_examples) < limit:
                removed_examples.append(line[:260])
            continue
        units = _sentence_units(raw_line)
        if not units:
            kept_lines.append(raw_line)
            continue
        kept_units: List[str] = []
        line_removed = 0
        for unit in units:
            stripped_unit = unit.strip()
            if stripped_unit and not CITATION_RE.search(stripped_unit) and text_has_factual_claim(stripped_unit):
                removed_count += 1
                line_removed += 1
                if len(removed_examples) < limit:
                    removed_examples.append(stripped_unit[:260])
                continue
            kept_units.append(unit)
        if line_removed and kept_units:
            kept_line = "".join(part.strip() for part in kept_units if str(part or "").strip()).strip()
            if kept_line:
                kept_lines.append(kept_line)
        elif not line_removed:
            kept_lines.append(raw_line)
    return "\n".join(kept_lines), {
        "citationless_factual_sentence_removed_count": removed_count,
        "citationless_factual_sentence_removed_examples": removed_examples,
    }


def finalize_markdown_citations(
    body_markdown: str,
    citation_manifest: Dict[str, Any],
    source_registry: Sequence[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Make final body citations and appendix sources an atomic pair.

    The manifest is built before all renderer/sanitizer passes, while the final
    body can still lose or gain numeric citation tokens. This step trusts the
    final body as the public surface, removes citations that cannot resolve to
    a traceable source, and renumbers the remaining body/appendix together.
    """

    original_body = str(body_markdown or "")
    lookup = _final_source_lookup(citation_manifest, source_registry)
    old_refs = _body_citation_refs(original_body)
    old_to_new: Dict[str, str] = {}
    appendix_sources: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    source_key_to_new: Dict[str, str] = {}

    def source_key(source: Dict[str, Any]) -> str:
        for key in ("url", "source_url", "document_ref", "document_id", "doc_id", "page_ref"):
            text = str(source.get(key) or "").strip().lower()
            if text:
                return f"{key}:{text}"
        aliases = sorted(_final_source_aliases(source))
        return "|".join(aliases)

    for old_ref in old_refs:
        source = lookup.get(old_ref) or lookup.get(_normalize_citation_ref(old_ref))
        if not source:
            unresolved.append(old_ref)
            continue
        key = source_key(source)
        new_ref = source_key_to_new.get(key)
        if not new_ref:
            new_ref = f"[{len(appendix_sources) + 1}]"
            source_key_to_new[key] = new_ref
            copied = dict(source)
            original_ref = str(copied.get("ref") or "").strip()
            if original_ref and original_ref != new_ref:
                copied.setdefault("original_ref", original_ref)
            copied["ref"] = new_ref
            copied["source_id"] = f"SRC-{len(appendix_sources) + 1:03d}"
            appendix_sources.append(copied)
        old_to_new[old_ref] = new_ref

    def replace(match: re.Match[str]) -> str:
        old_ref = f"[{match.group(1)}]"
        return old_to_new.get(old_ref, "")

    rewritten_body = re.sub(r"\[(\d{1,5})\]", replace, original_body)
    rewritten_body, duplicate_removed_count = _collapse_adjacent_duplicate_citations(rewritten_body)
    rewritten_body, bullet_drop_diagnostics = _drop_citationless_factual_bullets(rewritten_body)
    rewritten_body, short_line_drop_diagnostics = _drop_short_citationless_factual_lines(rewritten_body)
    rewritten_body, sentence_drop_diagnostics = _drop_citationless_factual_sentences(rewritten_body)
    final_body_refs = _body_citation_refs(rewritten_body)
    final_appendix_refs = [str(source.get("ref") or "").strip() for source in appendix_sources if str(source.get("ref") or "").strip()]
    missing_appendix_refs = [ref for ref in final_body_refs if ref not in set(final_appendix_refs)]
    citationless_examples = _citationless_factual_segments(rewritten_body)
    reconciliation_status = "blocked" if (missing_appendix_refs or citationless_examples) else "ok"
    citationless_removed_count = (
        int(bullet_drop_diagnostics.get("citationless_factual_bullet_removed_count") or 0)
        + int(short_line_drop_diagnostics.get("citationless_short_factual_line_removed_count") or 0)
        + int(sentence_drop_diagnostics.get("citationless_factual_sentence_removed_count") or 0)
    )
    rebind_threshold = _env_int(
        "REPORT_FINAL_CITATION_REBIND_REMOVAL_THRESHOLD",
        5,
        min_value=0,
        max_value=1000,
    )
    citation_rebind_required = bool(citationless_removed_count > rebind_threshold)
    diagnostics = {
        "final_citation_reconciliation_status": reconciliation_status,
        "final_body_citation_refs": final_body_refs,
        "final_appendix_refs": final_appendix_refs,
        "final_missing_appendix_refs": missing_appendix_refs,
        "final_unresolved_citation_removed_count": len(unresolved),
        "final_unresolved_citation_refs": unresolved,
        "final_duplicate_citation_removed_count": duplicate_removed_count,
        **bullet_drop_diagnostics,
        **short_line_drop_diagnostics,
        **sentence_drop_diagnostics,
        "citationless_factual_removed_count": citationless_removed_count,
        "citation_rebind_required": citation_rebind_required,
        "citation_rebind_reason": (
            "citationless_factual_removal_exceeded_threshold" if citation_rebind_required else ""
        ),
        "clean_report_eligible": not citation_rebind_required and reconciliation_status == "ok",
        "factual_body_without_citations_count": len(citationless_examples),
        "citationless_fact_examples": citationless_examples,
    }
    return rewritten_body, appendix_sources, diagnostics


def _split_rendered_source_appendix(markdown: str) -> tuple[str, str]:
    lines = str(markdown or "").splitlines()
    appendix_start: Optional[int] = None
    for index, line in enumerate(lines):
        if not re.match(r"^##\s+", line):
            continue
        if any(token in line for token in ("来源", "附录", "Source", "Appendix", "鏉ユ簮", "闄勫綍")):
            appendix_start = index
            break
    if appendix_start is None:
        return str(markdown or "").strip(), ""
    return "\n".join(lines[:appendix_start]).strip(), "\n".join(lines[appendix_start:]).strip()


def _rewrite_final_markdown_with_reconciled_appendix(
    markdown: str,
    *,
    citation_manifest: Dict[str, Any],
    source_registry: Sequence[Dict[str, Any]],
    appendix_package: Dict[str, Any],
) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    body, _old_appendix = _split_rendered_source_appendix(markdown)
    body, reconciled_sources, diagnostics = finalize_markdown_citations(body, citation_manifest, source_registry)
    parts = [body]
    if CITATION_RE.search(body) and reconciled_sources:
        rendered = _render_appendix_block(
            title=GLOBAL_BLOCK_TITLES.get("appendix", "appendix"),
            appendix_package=appendix_package,
            source_registry=reconciled_sources,
            rendered_groups=set(),
        )
        if rendered:
            parts.append(rendered)
    return "\n\n".join(part for part in parts if str(part or "").strip()), reconciled_sources, diagnostics


_METRIC_CLAIM_RE = re.compile(
    r"(?:CAGR|compound annual growth|market size|forecast|预计|预测|市场规模|复合增长率|年复合增长率|亿美元|亿元|万亿|203[0-9]年|\d+(?:\.\d+)?%)",
    re.I,
)
_STRONG_CLAIM_RE = re.compile(r"(?:market is broad|strong|huge|确定|显著|广阔|巨大|强结论|强支撑|确定性|高确定)", re.I)
_HARD_METRIC_CLAIM_RE = re.compile(
    r"(?:"
    r"CAGR|compound annual growth|forecast|"
    r"market size[^.\n。]{0,80}\d|"
    r"\d+(?:\.\d+)?\s*(?:%|亿元|亿美元|万亿|billion|million|trillion)|"
    r"20[2-4]\d\s*年[^.\n。]{0,40}(?:达到|达|为|增长|规模|预算|CAGR)"
    r")",
    re.I,
)
_WEAK_SOURCE_LEVELS = {"c", "d", "search_result_only", "inaccessible"}
_ENTITY_RE = re.compile(r"\b(?:OpenAI|Google|Microsoft|Salesforce|IDC|Gartner|AWS|Anthropic|Alibaba|Baidu|Tencent)\b|阿里|百度|腾讯|华为|字节", re.I)
_SINGLE_COMPANY_SOURCE_RE = re.compile(
    r"(?:investor\s*relations|ircs|questionDetail|q&a|question\s*and\s*answer|"
    r"company\s+(?:q&a|faq)|official\s+(?:q&a|faq)|"
    r"投资者关系|互动易|问答|公司问答|官方问答|公司官方|产品问答)",
    re.I,
)
_BROAD_MARKET_CLAIM_RE = re.compile(
    r"(?:"
    r"\b(?:field|industry|market|ecosystem|sector|overall|across\s+the|broad(?:ly)?)\b|"
    r"formed\s+verifiable\s+demand|verifiable\s+market|market\s+space|"
    r"领域|行业|生态|市场|整体|全场景|全行业|普遍|已形成|真实需求|可验证需求|市场空间|商业化潜力"
    r")",
    re.I,
)
_SOFT_SOURCE_CLAIM_REASONS = {"weak_source_strong_claim", "entity_not_supported_by_evidence"}
_METRIC_VALUE_TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|亿|万|万元|亿元|亿美元|人民币|CNY|RMB|billion|million|trillion|yuan)",
    re.I,
)


def _source_claim_gate_mode() -> str:
    value = str(os.environ.get("REPORT_SOURCE_CLAIM_GATE_MODE") or "balanced").strip().lower()
    if value in {"strict", "clean"}:
        return "strict"
    if value in {"relaxed", "balanced"}:
        return "balanced"
    return "balanced"


def _support_blob(section: Dict[str, Any]) -> str:
    values: List[str] = []
    for key in ("claim", "reasoning", "mechanism", "counter_evidence", "section_title"):
        values.append(str(section.get(key) or ""))
    for key in ("supporting_facts", "evidence_basis"):
        for item in _as_list(section.get(key)):
            if isinstance(item, dict):
                values.append(" ".join(str(item.get(field) or "") for field in ("distilled_fact", "source_title", "title", "value", "unit", "period", "time_or_scope")))
            else:
                values.append(str(item or ""))
    for block in _as_list(section.get("render_blocks")):
        if isinstance(block, dict):
            values.append(str(block.get("text") or ""))
    return " ".join(values)


def _supporting_fact_text(section: Dict[str, Any], *, max_chars: int = 180) -> str:
    for key in ("supporting_facts", "evidence_basis", "fact_cards"):
        for item in _as_list(section.get(key)):
            if isinstance(item, dict):
                text = " ".join(
                    str(item.get(field) or "")
                    for field in ("distilled_fact", "public_fact_card", "action_or_signal", "source_title", "title")
                ).strip()
            else:
                text = str(item or "").strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                return text[:max_chars].rstrip()
    text = re.sub(r"\s+", " ", str(section.get("reasoning") or section.get("claim") or "").strip())
    return text[:max_chars].rstrip()


def _metric_fact_is_structured(item: Dict[str, Any]) -> bool:
    value = str(item.get("value") or item.get("metric_value") or "").strip()
    unit = str(item.get("unit") or item.get("metric_unit") or "").strip()
    period = str(item.get("period") or item.get("time_or_scope") or item.get("date") or "").strip()
    source_ref = str(
        item.get("source_ref")
        or item.get("citation_ref")
        or item.get("ref")
        or item.get("evidence_id")
        or ""
    ).strip()
    return bool(value and source_ref and (unit or period))


_AI_AGENT_TOPIC_RE = re.compile(r"AI\s*Agent(?:s)?|AIAgent(?:s)?|\bagentic\b|智能体|智能代理|数字员工", re.I)


def _metric_source_matches_section_topic(
    *,
    section: Dict[str, Any],
    source: Dict[str, Any] | None = None,
    metric_fact: Dict[str, Any] | None = None,
    topic_context: str = "",
) -> bool:
    section_blob = " ".join([_support_blob(section), str(topic_context or "")])
    if not _AI_AGENT_TOPIC_RE.search(section_blob):
        return True
    source_blob = ""
    if isinstance(source, dict):
        source_blob += " ".join(
            str(source.get(field) or "")
            for field in (
                "title",
                "source_title",
                "url",
                "source_url",
                "summary",
                "publisher",
                "domain",
            )
        )
    if isinstance(metric_fact, dict):
        source_blob += " " + " ".join(
            str(metric_fact.get(field) or "")
            for field in (
                "distilled_fact",
                "fact",
                "metric",
                "variable",
                "subject",
                "source_title",
                "source_url",
            )
        )
    return bool(_AI_AGENT_TOPIC_RE.search(source_blob))


def _section_has_structured_metric_fact(
    section: Dict[str, Any],
    source_registry: Sequence[Dict[str, Any]] | None = None,
    *,
    topic_context: str = "",
) -> bool:
    candidates: List[Dict[str, Any]] = []
    for key in ("supporting_facts", "evidence_basis", "fact_cards"):
        for item in _as_list(section.get(key)):
            if isinstance(item, dict):
                candidates.append(item)
    if any(str(section.get(key) or "").strip() for key in ("value", "metric_value")):
        candidates.append(section)
    for item in candidates:
        if _metric_fact_is_structured(item) and _metric_source_matches_section_topic(section=section, metric_fact=item, topic_context=topic_context):
            return True
    if source_registry:
        lookup = _final_source_lookup({}, source_registry)
        refs = [
            *_as_list(section.get("used_fact_refs")),
            *_as_list(section.get("evidence_refs")),
            *_as_list(section.get("citation_refs")),
            *_as_list(section.get("source_refs")),
        ]
        for ref in refs:
            text = str(ref or "").strip()
            if not text:
                continue
            source = lookup.get(text) or lookup.get(_normalize_citation_ref(text))
            if not source:
                continue
            for metric_fact in _as_list(source.get("metric_facts")):
                if (
                    isinstance(metric_fact, dict)
                    and _metric_fact_is_structured(metric_fact)
                    and _metric_source_matches_section_topic(section=section, source=source, metric_fact=metric_fact, topic_context=topic_context)
                ):
                    return True
            if _metric_fact_is_structured(source) and _metric_source_matches_section_topic(section=section, source=source, metric_fact=source, topic_context=topic_context):
                return True
    return False


def _source_level_for_refs(refs: Sequence[Any], source_registry: Sequence[Dict[str, Any]]) -> set[str]:
    lookup = _final_source_lookup({}, source_registry)
    levels: set[str] = set()
    for ref in refs:
        source = lookup.get(str(ref or "").strip()) or lookup.get(_normalize_citation_ref(ref))
        if not source:
            continue
        for value in (source.get("source_level"), source.get("source_verification_status")):
            text = str(value or "").strip().lower()
            if text:
                levels.add(text)
    return levels


def _sources_for_section_refs(section: Dict[str, Any], source_registry: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = _final_source_lookup({}, source_registry)
    refs = [
        *_as_list(section.get("used_fact_refs")),
        *_as_list(section.get("evidence_refs")),
        *_as_list(section.get("citation_refs")),
        *_as_list(section.get("source_refs")),
    ]
    sources: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        text = str(ref or "").strip()
        if not text:
            continue
        source = lookup.get(text) or lookup.get(_normalize_citation_ref(text))
        if not isinstance(source, dict):
            continue
        key = str(source.get("url") or source.get("source_url") or source.get("ref") or source.get("evidence_id") or text)
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)
    return sources


def _metric_value_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _METRIC_VALUE_TOKEN_RE.finditer(str(text or "")):
        token = re.sub(r"\s+", "", match.group(0).lower())
        if token:
            tokens.add(token)
    return tokens


def _source_metric_support_blob(sources: Sequence[Dict[str, Any]]) -> str:
    values: List[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        values.extend(
            str(source.get(field) or "")
            for field in (
                "title",
                "source_title",
                "url",
                "source_url",
                "summary",
                "quote",
                "value",
                "unit",
                "period",
                "date",
            )
        )
        for metric_fact in _as_list(source.get("metric_facts")):
            if isinstance(metric_fact, dict):
                values.extend(
                    str(metric_fact.get(field) or "")
                    for field in (
                        "distilled_fact",
                        "fact",
                        "metric",
                        "value",
                        "unit",
                        "period",
                        "time_or_scope",
                    )
                )
    return " ".join(values)


def _metric_fact_ref_mismatch(section: Dict[str, Any], source_registry: Sequence[Dict[str, Any]]) -> bool:
    if str(section.get("block_type") or "").strip() != "metric_reconciliation":
        return False
    support_tokens = _metric_value_tokens(_support_blob(section))
    if not support_tokens:
        return False
    source_tokens = _metric_value_tokens(_source_metric_support_blob(_sources_for_section_refs(section, source_registry)))
    if not source_tokens:
        return True
    return not support_tokens.issubset(source_tokens)


def _single_company_source_overbroad_claim(section: Dict[str, Any], source_registry: Sequence[Dict[str, Any]]) -> bool:
    sources = _sources_for_section_refs(section, source_registry)
    if len(sources) != 1:
        return False
    source_text = " ".join(
        str(sources[0].get(key) or "")
        for key in ("title", "source_title", "url", "source_url", "summary", "publisher", "domain")
    )
    if not _SINGLE_COMPANY_SOURCE_RE.search(source_text):
        return False
    claim_text = " ".join(str(section.get(key) or "") for key in ("claim", "section_title"))
    for block in _as_list(section.get("render_blocks")):
        if isinstance(block, dict):
            claim_text += " " + str(block.get("text") or "")
    return bool(_BROAD_MARKET_CLAIM_RE.search(claim_text))


def _narrow_single_company_overbroad_section(section: Dict[str, Any]) -> Dict[str, Any]:
    support_text = _support_blob(section)
    use_chinese = bool(re.search(r"[\u4e00-\u9fff]", support_text))
    updated = dict(section)
    updated["claim_strength"] = "directional"
    updated["source_claim_demoted_from"] = "single_company_source_overbroad_claim"
    if use_chinese:
        title = "教育场景的单点落地信号"
        claim = "公司问答披露了一个教育场景的 AI Agent 部署样本。"
        reasoning = (
            "这个样本说明产品功能已经进入具体教学、学习或管理流程，"
            "更适合作为单点落地信号；结论边界是尚不能代表更大范围的需求规模。"
        )
        paragraph = f"{claim}{reasoning}"
    else:
        title = "Single deployment case"
        claim = "The company Q&A supports one AI Agent deployment case."
        reasoning = (
            "It can be used as a single deployment case signal, but it does not prove demand across the wider field."
        )
        paragraph = f"{claim} {reasoning}"
    updated["section_title"] = title
    updated["claim"] = claim
    updated["reasoning"] = reasoning
    updated["mechanism"] = ""
    updated["counter_evidence"] = ""
    render_blocks: List[Dict[str, Any]] = []
    paragraph_written = False
    for block in _as_list(section.get("render_blocks")):
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "paragraph").lower() == "paragraph" and not paragraph_written:
            copied_block = dict(block)
            copied_block["text"] = paragraph
            render_blocks.append(copied_block)
            paragraph_written = True
            continue
        if str(block.get("type") or "paragraph").lower() != "paragraph":
            render_blocks.append(dict(block))
    if not paragraph_written:
        render_blocks.insert(0, {"type": "paragraph", "text": paragraph})
    updated["render_blocks"] = render_blocks
    return updated


def _narrow_soft_source_claim_section(section: Dict[str, Any], reason: str) -> Dict[str, Any]:
    support_text = _supporting_fact_text(section)
    use_chinese = bool(re.search(r"[\u4e00-\u9fff]", support_text))
    updated = dict(section)
    updated["claim_strength"] = "directional"
    updated["source_claim_demoted_from"] = reason
    updated["source_claim_gate_action"] = "demote"
    if use_chinese:
        fact = support_text or "已引用材料"
        title = "方向性证据信号"
        claim = f"{fact}这一材料更适合作为方向性信号。"
        reasoning = "它可以说明局部场景中已有可观察动作，但不能直接外推为强市场结论。"
        paragraph = f"{claim}{reasoning}"
    else:
        fact = support_text or "The cited material"
        title = "Directional evidence signal"
        claim = f"{fact} supports a directional signal."
        reasoning = "It should be limited to the cited scenario rather than used as a strong market-wide conclusion."
        paragraph = f"{claim} {reasoning}"
    updated["section_title"] = title
    updated["claim"] = claim
    updated["reasoning"] = reasoning
    updated["mechanism"] = ""
    updated["counter_evidence"] = ""
    render_blocks: List[Dict[str, Any]] = []
    paragraph_written = False
    for block in _as_list(section.get("render_blocks")):
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "paragraph").lower()
        if block_type == "paragraph" and not paragraph_written:
            copied_block = dict(block)
            copied_block["text"] = paragraph
            render_blocks.append(copied_block)
            paragraph_written = True
            continue
        if block_type != "paragraph":
            render_blocks.append(dict(block))
    if not paragraph_written:
        render_blocks.insert(0, {"type": "paragraph", "text": paragraph})
    updated["render_blocks"] = render_blocks
    return updated


def _section_source_claim_mismatch_reason(section: Dict[str, Any], source_registry: Sequence[Dict[str, Any]], *, topic_context: str = "") -> str:
    text = _support_blob(section)
    block_type = str(section.get("block_type") or "").strip()
    refs = [*_as_list(section.get("used_fact_refs")), *_as_list(section.get("evidence_refs")), *_as_list(section.get("citation_refs"))]
    metric_claim_match = (
        _METRIC_CLAIM_RE.search(text)
        if block_type == "metric_reconciliation"
        else _HARD_METRIC_CLAIM_RE.search(text)
    )
    if metric_claim_match:
        if _metric_fact_ref_mismatch(section, source_registry):
            return "metric_fact_ref_mismatch"
        if not _section_has_structured_metric_fact(section, source_registry, topic_context=topic_context) and (
            refs or source_registry or section.get("unresolved_refs_filtered")
        ):
            return "metric_claim_without_metric_fact"
    source_levels = _source_level_for_refs(refs, source_registry)
    if _STRONG_CLAIM_RE.search(text) and bool(source_levels.intersection(_WEAK_SOURCE_LEVELS)):
        return "weak_source_strong_claim"
    if _single_company_source_overbroad_claim(section, source_registry):
        return "single_company_source_overbroad_claim"
    entities = {match.group(0).lower() for match in _ENTITY_RE.finditer(str(section.get("claim") or ""))}
    if entities:
        evidence_text = " ".join(
            str(value or "")
            for value in [
                section.get("reasoning"),
                section.get("mechanism"),
                *list(_as_list(section.get("supporting_facts"))),
                *list(_as_list(section.get("evidence_basis"))),
            ]
        ).lower()
        unsupported = [entity for entity in entities if entity not in evidence_text]
        if unsupported:
            return "entity_not_supported_by_evidence"
    return ""


def _apply_source_claim_support_gate(
    chapters: Sequence[Dict[str, Any]],
    source_registry: Sequence[Dict[str, Any]],
    *,
    topic_context: str = "",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    diagnostics = {
        "source_claim_support_status": "ok",
        "source_gate_mode": _source_claim_gate_mode(),
        "section_dropped_due_to_source_claim_mismatch_count": 0,
        "section_dropped_due_to_unresolved_refs_count": 0,
        "factual_section_without_resolved_ref_count": 0,
        "empty_chapter_omitted_after_source_gate_count": 0,
        "source_claim_mismatch_examples": [],
        "citationless_fact_examples": [],
        "metric_claim_without_metric_fact_count": 0,
        "weak_source_strong_claim_demoted_count": 0,
        "demoted_section_count": 0,
        "hard_dropped_section_count": 0,
        "soft_gate_rewritten_count": 0,
        "relaxed_section_examples": [],
    }
    has_section_refs = any(
        isinstance(section, dict)
        and (
            _as_list(section.get("used_fact_refs"))
            or _as_list(section.get("evidence_refs"))
            or _as_list(section.get("citation_refs"))
            or section.get("unresolved_refs_filtered")
        )
        for chapter in list(chapters or [])
        if isinstance(chapter, dict)
        for section in _as_list(chapter.get("sections"))
    )
    if not any(isinstance(source, dict) for source in list(source_registry or [])) and not has_section_refs:
        diagnostics["source_claim_support_status"] = "skipped_no_source_registry"
        return [dict(chapter) for chapter in list(chapters or []) if isinstance(chapter, dict)], diagnostics
    gated_chapters: List[Dict[str, Any]] = []
    for chapter in list(chapters or []):
        if not isinstance(chapter, dict):
            continue
        copied = dict(chapter)
        kept_sections: List[Dict[str, Any]] = []
        for section in _as_list(chapter.get("sections")):
            if not isinstance(section, dict):
                continue
            lineage = resolve_section_source_refs(section, source_registry)
            reason = ""
            if lineage.get("has_factual_claim") and not lineage.get("has_resolved_ref"):
                reason = "factual_section_without_resolved_ref"
            if not reason:
                kept_sections.append(section)
                continue
            if reason == "single_company_source_overbroad_claim":
                kept_sections.append(_narrow_single_company_overbroad_section(section))
                diagnostics["weak_source_strong_claim_demoted_count"] += 1
                diagnostics["demoted_section_count"] += 1
                diagnostics["soft_gate_rewritten_count"] += 1
                if len(diagnostics["source_claim_mismatch_examples"]) < 8:
                    diagnostics["source_claim_mismatch_examples"].append(
                        {
                            "chapter_id": chapter.get("chapter_id"),
                            "section_id": section.get("section_id"),
                            "block_type": section.get("block_type"),
                            "reason": reason,
                            "claim": str(section.get("claim") or "")[:220],
                        }
                    )
                if len(diagnostics["relaxed_section_examples"]) < 8:
                    diagnostics["relaxed_section_examples"].append(
                        {
                            "chapter_id": chapter.get("chapter_id"),
                            "section_id": section.get("section_id"),
                            "block_type": section.get("block_type"),
                            "reason": reason,
                            "action": "demoted",
                            "claim": str(section.get("claim") or "")[:220],
                        }
                    )
                continue
            if (
                diagnostics["source_gate_mode"] != "strict"
                and reason in _SOFT_SOURCE_CLAIM_REASONS
                and lineage.get("has_resolved_ref")
            ):
                kept_sections.append(_narrow_soft_source_claim_section(section, reason))
                diagnostics["demoted_section_count"] += 1
                diagnostics["soft_gate_rewritten_count"] += 1
                if reason == "weak_source_strong_claim":
                    diagnostics["weak_source_strong_claim_demoted_count"] += 1
                if len(diagnostics["source_claim_mismatch_examples"]) < 8:
                    diagnostics["source_claim_mismatch_examples"].append(
                        {
                            "chapter_id": chapter.get("chapter_id"),
                            "section_id": section.get("section_id"),
                            "block_type": section.get("block_type"),
                            "reason": reason,
                            "claim": str(section.get("claim") or "")[:220],
                        }
                    )
                if len(diagnostics["relaxed_section_examples"]) < 8:
                    diagnostics["relaxed_section_examples"].append(
                        {
                            "chapter_id": chapter.get("chapter_id"),
                            "section_id": section.get("section_id"),
                            "block_type": section.get("block_type"),
                            "reason": reason,
                            "action": "demoted",
                            "claim": str(section.get("claim") or "")[:220],
                        }
                    )
                continue
            diagnostics["section_dropped_due_to_source_claim_mismatch_count"] += 1
            diagnostics["hard_dropped_section_count"] += 1
            if reason == "factual_section_without_resolved_ref":
                diagnostics["section_dropped_due_to_unresolved_refs_count"] += 1
                diagnostics["factual_section_without_resolved_ref_count"] += 1
                if _METRIC_CLAIM_RE.search(_support_blob(section)):
                    diagnostics["metric_claim_without_metric_fact_count"] += 1
            if reason in {"metric_claim_without_metric_fact", "metric_fact_ref_mismatch"}:
                diagnostics["metric_claim_without_metric_fact_count"] += 1
            if reason == "weak_source_strong_claim":
                diagnostics["weak_source_strong_claim_demoted_count"] += 1
            if reason == "factual_section_without_resolved_ref" and len(diagnostics["citationless_fact_examples"]) < 8:
                diagnostics["citationless_fact_examples"].append(
                    {
                        "chapter_id": chapter.get("chapter_id"),
                        "section_id": section.get("section_id"),
                        "claim": str(section.get("claim") or section.get("reasoning") or "")[:220],
                    }
                )
            if len(diagnostics["source_claim_mismatch_examples"]) < 8:
                diagnostics["source_claim_mismatch_examples"].append(
                    {
                        "chapter_id": chapter.get("chapter_id"),
                        "section_id": section.get("section_id"),
                        "block_type": section.get("block_type"),
                        "reason": reason,
                        "claim": str(section.get("claim") or "")[:220],
                    }
                )
        copied["sections"] = kept_sections
        gated_chapters.append(copied)
    if diagnostics["section_dropped_due_to_source_claim_mismatch_count"]:
        diagnostics["source_claim_support_status"] = "filtered"
    elif diagnostics["weak_source_strong_claim_demoted_count"]:
        diagnostics["source_claim_support_status"] = "demoted"
    return gated_chapters, diagnostics


def _drop_factual_sections_without_manifest_citations(
    chapters: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    diagnostics = {
        "section_dropped_due_to_source_claim_mismatch_count": 0,
        "section_dropped_due_to_unresolved_refs_count": 0,
        "factual_section_without_resolved_ref_count": 0,
        "source_claim_mismatch_examples": [],
        "citationless_fact_examples": [],
    }
    gated_chapters: List[Dict[str, Any]] = []
    for chapter in list(chapters or []):
        if not isinstance(chapter, dict):
            continue
        copied = dict(chapter)
        kept_sections: List[Dict[str, Any]] = []
        for section in _as_list(chapter.get("sections")):
            if not isinstance(section, dict):
                continue
            support_blob = _support_blob(section)
            requires_citation = bool(
                text_has_factual_claim(support_blob)
                or section.get("evidence_backed")
                or _as_list(section.get("used_fact_refs"))
                or _as_list(section.get("evidence_refs"))
            )
            if requires_citation and not _as_list(section.get("citation_refs")):
                diagnostics["section_dropped_due_to_source_claim_mismatch_count"] += 1
                diagnostics["section_dropped_due_to_unresolved_refs_count"] += 1
                diagnostics["factual_section_without_resolved_ref_count"] += 1
                if len(diagnostics["citationless_fact_examples"]) < 8:
                    diagnostics["citationless_fact_examples"].append(
                        {
                            "chapter_id": chapter.get("chapter_id"),
                            "section_id": section.get("section_id"),
                            "claim": str(section.get("claim") or section.get("reasoning") or "")[:220],
                        }
                    )
                if len(diagnostics["source_claim_mismatch_examples"]) < 8:
                    diagnostics["source_claim_mismatch_examples"].append(
                        {
                            "chapter_id": chapter.get("chapter_id"),
                            "section_id": section.get("section_id"),
                            "block_type": section.get("block_type"),
                            "reason": "manifest_citation_missing",
                            "claim": str(section.get("claim") or "")[:220],
                        }
                    )
                continue
            kept_sections.append(section)
        copied["sections"] = kept_sections
        gated_chapters.append(copied)
    return gated_chapters, diagnostics


def _merge_source_claim_support_diagnostics(target: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(target or {})
    for key in (
        "section_dropped_due_to_source_claim_mismatch_count",
        "section_dropped_due_to_unresolved_refs_count",
        "factual_section_without_resolved_ref_count",
        "empty_chapter_omitted_after_source_gate_count",
        "metric_claim_without_metric_fact_count",
        "weak_source_strong_claim_demoted_count",
        "demoted_section_count",
        "hard_dropped_section_count",
        "soft_gate_rewritten_count",
    ):
        merged[key] = int(merged.get(key) or 0) + int(extra.get(key) or 0)
    for key in ("source_claim_mismatch_examples", "citationless_fact_examples", "relaxed_section_examples"):
        merged[key] = _as_list(merged.get(key)) + _as_list(extra.get(key))
        if len(merged[key]) > 8:
            merged[key] = merged[key][:8]
    merged.setdefault("source_gate_mode", target.get("source_gate_mode") or extra.get("source_gate_mode") or _source_claim_gate_mode())
    if int(extra.get("section_dropped_due_to_source_claim_mismatch_count") or 0):
        merged["source_claim_support_status"] = "filtered"
    elif int(merged.get("demoted_section_count") or 0) or int(merged.get("weak_source_strong_claim_demoted_count") or 0):
        merged["source_claim_support_status"] = "demoted"
    else:
        merged.setdefault("source_claim_support_status", target.get("source_claim_support_status", "ok"))
    return merged


def _claim_refs_for_transfer(item: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for key in ("used_fact_refs", "evidence_refs", "supporting_fact_refs", "used_evidence_ids", "supporting_evidence_refs"):
        for value in _as_list(item.get(key)):
            if isinstance(value, dict):
                ref = value.get("evidence_id") or value.get("ref") or value.get("source_ref") or value.get("id")
            else:
                ref = value
            if str(ref or "").strip():
                refs.append(str(ref).strip())
    return _dedupe_strings(refs, limit=50)


def _is_analysis_claim_for_transfer(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    if not str(item.get("claim") or item.get("judgment") or item.get("takeaway") or "").strip():
        return False
    if not _claim_refs_for_transfer(item):
        return False
    if item.get("claim_id") or item.get("source_support_map") or item.get("analysis_role"):
        return True
    strength = str(item.get("claim_strength") or "").strip().lower()
    return strength in {"strong", "moderate", "directional", "contextual", "limited_evidence"}


def _iter_public_sections(chapters: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for chapter in chapters or []:
        if not isinstance(chapter, dict):
            continue
        for section in _as_list(chapter.get("sections")):
            if isinstance(section, dict):
                yield section


def _analysis_transfer_diagnostics(
    *,
    claim_units: Sequence[Dict[str, Any]],
    public_chapters: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    analysis_claims = [
        dict(item)
        for item in claim_units or []
        if isinstance(item, dict) and _is_analysis_claim_for_transfer(item)
    ]
    claim_ids = _dedupe_strings(
        [
            str(item.get("claim_id") or item.get("id") or "").strip()
            for item in analysis_claims
            if str(item.get("claim_id") or item.get("id") or "").strip()
        ],
        limit=200,
    )
    rendered_sections = [
        dict(section)
        for section in _iter_public_sections(public_chapters)
        if section.get("claim_id") or section.get("source_support_map") or section.get("analysis_role")
    ]
    rendered_ids = _dedupe_strings(
        [
            str(section.get("claim_id") or section.get("id") or "").strip()
            for section in rendered_sections
            if str(section.get("claim_id") or section.get("id") or "").strip()
        ],
        limit=200,
    )
    rendered_id_set = set(rendered_ids)
    rendered_ref_sets = [
        set(_claim_refs_for_transfer(section))
        for section in rendered_sections
        if _claim_refs_for_transfer(section)
    ]
    matched_ids: List[str] = []
    matched_by_ref_ids: List[str] = []
    lost_ids: List[str] = []
    for claim in analysis_claims:
        claim_id = str(claim.get("claim_id") or claim.get("id") or "").strip()
        claim_refs = set(_claim_refs_for_transfer(claim))
        matched_by_id = bool(claim_id and claim_id in rendered_id_set)
        matched_by_ref = bool(claim_id and claim_refs and any(claim_refs.intersection(refs) for refs in rendered_ref_sets))
        if matched_by_id or matched_by_ref:
            if claim_id:
                matched_ids.append(claim_id)
                if matched_by_ref and not matched_by_id:
                    matched_by_ref_ids.append(claim_id)
        elif claim_id:
            lost_ids.append(claim_id)
    claim_count = len(analysis_claims)
    matched_ids = _dedupe_strings(matched_ids, limit=200)
    matched_by_ref_ids = _dedupe_strings(matched_by_ref_ids, limit=200)
    lost_ids = _dedupe_strings(lost_ids, limit=200)
    rendered_claim_count = len(matched_ids)
    missing_id_count = max(0, claim_count - len(claim_ids))
    reason_counts: Dict[str, int] = {}
    if lost_ids:
        reason_counts["not_rendered_in_public_sections"] = len(lost_ids)
    if missing_id_count:
        reason_counts["missing_claim_id"] = missing_id_count
    return {
        "analysis_claim_count": claim_count,
        "analysis_claim_count_by_strength": dict(
            Counter(
                str(item.get("claim_strength") or "unknown").strip().lower() or "unknown"
                for item in analysis_claims
            )
        ),
        "rendered_analysis_section_count": len(rendered_sections),
        "rendered_analysis_claim_count": rendered_claim_count,
        "claim_to_section_transfer_rate": round(rendered_claim_count / claim_count, 3) if claim_count else 0.0,
        "claim_lost_after_analysis_count": max(0, claim_count - rendered_claim_count),
        "claim_lost_after_analysis_reasons": reason_counts,
        "analysis_claim_ids_input": claim_ids[:50],
        "analysis_claim_ids_rendered": matched_ids[:50],
        "analysis_claim_ids_rendered_by_ref": matched_by_ref_ids[:50],
        "analysis_claim_ids_lost": lost_ids[:50],
        "analysis_fact_usage_count": sum(len(_claim_refs_for_transfer(item)) for item in analysis_claims),
    }


def should_render_chapter(chapter: Dict[str, Any]) -> bool:
    if chapter.get("omit_from_report"):
        return False

    lead = _public_text(chapter.get("lead"))

    sections = [
        public_section
        for item in _as_list(chapter.get("sections"))
        if isinstance(item, dict)
        and not item.get("omit_from_report")
        for public_section in [_public_section(item)]
        if _section_has_public_content(public_section)
    ]
    tables = [
        item
        for item in _as_list(chapter.get("table_packages"))
        if isinstance(item, dict) and _table_passed_for_public(item)
    ]
    return bool(lead or sections or tables)


def _chapter_has_public_body(chapter: Dict[str, Any]) -> bool:
    sections = [
        section
        for section in _as_list(chapter.get("sections"))
        if isinstance(section, dict)
        and not section.get("omit_from_report")
        and _section_has_public_content(section)
    ]
    tables = [
        item
        for item in _as_list(chapter.get("table_packages"))
        if isinstance(item, dict) and _table_passed_for_public(item) and str(render_table_package(item) or "").strip()
    ]
    return bool(sections or tables)


def _public_chapter(chapter: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(chapter)
    copied["sections"] = [
        public_section
        for section in _as_list(chapter.get("sections"))
        if isinstance(section, dict)
        and not section.get("omit_from_report")
        for public_section in [_public_section(section)]
        if _section_has_public_content(public_section)
    ]
    copied["table_packages"] = [
        table
        for table in _as_list(chapter.get("table_packages"))
        if isinstance(table, dict) and _table_passed_for_public(table)
    ]
    copied["lead"] = _public_text(copied.get("lead"))
    return copied


def should_render_key_data(table_packages: Sequence[Dict[str, Any]]) -> bool:
    return any(
        isinstance(table, dict)
        and _table_passed_for_public(table)
        for table in list(table_packages or [])
    )


def _as_block_list(value: Any, fallback: Sequence[str]) -> List[str]:
    blocks = [str(item or "").strip() for item in _as_list(value) if str(item or "").strip()]
    return blocks or list(fallback)


def _rename_first_h2(markdown: str, title: str) -> str:
    markdown = str(markdown or "").strip()
    title = str(title or "").strip()
    if not markdown or not title:
        return markdown
    if markdown.startswith("## "):
        return markdown.replace(markdown.splitlines()[0], f"## {title}", 1)
    return f"## {title}\n{markdown}"


def _render_key_data_block(title: str, decision_package: Dict[str, Any], table_packages: Sequence[Dict[str, Any]]) -> str:
    del decision_package

    def row_citation_suffix(row: Dict[str, Any]) -> str:
        refs: List[str] = []
        for key in ("citation_refs", "source_refs"):
            refs.extend(_as_list(row.get(key)))
        for key in ("source_ref", "citation_ref", "ref"):
            value = str(row.get(key) or "").strip()
            if value:
                refs.append(value)
        public_refs = [
            ref
            for ref in (_normalize_citation_ref(item) for item in refs)
            if re.fullmatch(r"\[\d{1,5}\]", ref)
        ]
        return "".join(_dedupe_strings(public_refs)[:2])

    rows: List[str] = []
    for table in table_packages:
        table = _as_dict(table)
        if not _table_passed_for_public(table):
            continue
        headers = _as_list(table.get("headers"))
        for row in _as_list(table.get("rows"))[:3]:
            row = _as_dict(row)
            suffix = row_citation_suffix(row)
            if not suffix:
                continue
            text = _key_data_bullet_from_table_row(headers, row)
            if text:
                if not re.search(r"(?:\[\d{1,5}\])+\s*$", text):
                    text = text.rstrip("銆傦紱; ") + " " + suffix
                rows.append(text)
    if not rows:
        return ""
    return "\n".join([f"## {title}", *[f"- {item}" for item in rows[:6]]])


def _render_watchlist_block(title: str, decision_package: Dict[str, Any]) -> str:
    rows: List[str] = []
    for item in _as_list(decision_package.get("watchlist"))[:8]:
        metric = _public_text(_as_dict(item).get("metric"))
        if metric:
            rows.append(f"- {metric}")
    for item in _as_list(decision_package.get("abandon_conditions"))[:4]:
        condition = _public_text(_as_dict(item).get("condition"))
        if condition:
            rows.append(f"- 反证：{condition}")
    if not rows:
        return ""
    return "\n".join([f"## {title}", *rows])


def _render_global_block(
    block_key: str,
    *,
    title_override: str = "",
    decision_package: Dict[str, Any],
    risk_package: Dict[str, Any],
    appendix_package: Dict[str, Any],
    source_registry: Sequence[Dict[str, Any]],
    table_packages: Sequence[Dict[str, Any]],
    rendered_groups: set[str],
) -> str:
    key = str(block_key or "").strip()
    title = title_override or GLOBAL_BLOCK_TITLES.get(key, key.replace("_", " ").strip().title())
    if key in SUMMARY_BLOCKS:
        if "summary" in rendered_groups:
            return ""
        rendered_groups.add("summary")
        rendered = _rename_first_h2(render_executive_summary(decision_package, table_packages), title)
        if re.search(r"(?m)^##\s*关键数据(?:\s|$)", rendered):
            rendered_groups.add("key_data")
        return rendered
    if key == "key_data":
        if "key_data" in rendered_groups:
            return ""
        rendered_groups.add("key_data")
        return _render_key_data_block(title, decision_package, table_packages)
    if key in DECISION_BLOCKS:
        if "decision" in rendered_groups:
            return ""
        rendered_groups.add("decision")
        return _rename_first_h2(render_decision_package(decision_package), title)
    if key in RISK_BLOCKS:
        if "risk" in rendered_groups:
            return ""
        rendered_groups.add("risk")
        return _rename_first_h2(render_risk_package(risk_package), title)
    if key in WATCHLIST_BLOCKS:
        if "watchlist" in rendered_groups:
            return ""
        rendered_groups.add("watchlist")
        return _render_watchlist_block(title, decision_package)
    if key == "appendix":
        if not _source_appendix_enabled():
            return ""
        return _render_appendix_block(
            title=title,
            appendix_package=appendix_package,
            source_registry=source_registry,
            rendered_groups=rendered_groups,
        )
    return ""


def _render_appendix_block(
    *,
    title: str,
    appendix_package: Dict[str, Any],
    source_registry: Sequence[Dict[str, Any]],
    rendered_groups: set[str],
) -> str:
    if "appendix" in rendered_groups:
        return ""
    rendered = render_appendix(source_registry, appendix_package)
    if not str(rendered or "").strip():
        return ""
    rendered_groups.add("appendix")
    return _rename_first_h2(rendered, title)


def run_final_writer_agent(
    *,
    query: str = "",
    report_blueprint: Optional[Dict[str, Any]] = None,
    chapter_packages: Optional[Sequence[Dict[str, Any]]] = None,
    table_packages: Optional[Sequence[Dict[str, Any]]] = None,
    decision_package: Optional[Dict[str, Any]] = None,
    risk_package: Optional[Dict[str, Any]] = None,
    appendix_package: Optional[Dict[str, Any]] = None,
    source_registry: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
    chapter_evidence_packages: Optional[Sequence[Dict[str, Any]]] = None,
    claim_units: Optional[Sequence[Dict[str, Any]]] = None,
    analysis_claim_units: Optional[Sequence[Dict[str, Any]]] = None,
    analysis_stage_diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_blueprint = _as_dict(report_blueprint)
    chapter_packages = [item for item in list(chapter_packages or []) if isinstance(item, dict)]
    public_chapters = [_public_chapter(chapter) for chapter in chapter_packages if should_render_chapter(chapter)]
    table_packages = _all_table_packages(public_chapters, [item for item in list(table_packages or []) if isinstance(item, dict)])
    table_packages = [
        table
        for table in table_packages
        if isinstance(table, dict) and _table_passed_for_public(table)
    ]
    decision_package = _as_dict(decision_package)
    decision_package = {
        **decision_package,
        "chapter_syntheses": [
            {
                "chapter_title": chapter.get("chapter_title"),
                "chapter_question": chapter.get("chapter_question"),
                "chapter_summary": _as_dict(chapter.get("chapter_summary")),
            }
            for chapter in public_chapters
            if _as_dict(chapter.get("chapter_summary"))
        ],
    }
    risk_package = _as_dict(risk_package)
    appendix_package = _as_dict(appendix_package)
    evidence_package_dict = _as_dict(evidence_package)
    full_source_registry = merge_source_registries(
        [item for item in list(source_registry or []) if isinstance(item, dict)],
        [item for item in _as_list(evidence_package_dict.get("source_registry")) if isinstance(item, dict)],
        [item for item in _as_list(evidence_package_dict.get("sources")) if isinstance(item, dict)],
        evidence_source_entries_from_package(
            evidence_package=evidence_package_dict,
            chapter_evidence_packages=[item for item in list(chapter_evidence_packages or []) if isinstance(item, dict)],
        ),
    )
    source_registry, excluded_sources = _traceable_source_registry(
        [item for item in list(full_source_registry or []) if isinstance(item, dict)],
        query=query,
    )
    manifest_claim_units = [
        *[item for item in list(claim_units or []) if isinstance(item, dict)],
        *[item for key in ("claim_units", "key_claims", "key_judgments") for item in _as_list(decision_package.get(key)) if isinstance(item, dict)],
    ]
    public_chapters, manifest_claim_units, ref_lineage_diagnostics = _filter_rendered_refs_with_source_registry(
        chapters=public_chapters,
        claim_units=manifest_claim_units,
        source_registry=full_source_registry,
    )
    topic_context = " ".join(
        str(value or "")
        for value in (
            query,
            report_blueprint.get("report_title"),
            report_blueprint.get("research_object"),
            report_blueprint.get("planning_query"),
        )
    )
    public_chapters, source_claim_support = _apply_source_claim_support_gate(
        public_chapters,
        source_registry,
        topic_context=topic_context,
    )
    citation_manifest = build_citation_manifest(
        chapters=public_chapters,
        claim_units=manifest_claim_units,
        source_registry=source_registry,
    )
    citation_manifest["filtered_unresolved_ref_reasons"] = _filtered_unresolved_ref_reason_details(
        _as_list(citation_manifest.get("filtered_unresolved_refs")),
        full_source_registry=full_source_registry,
        excluded_sources=excluded_sources,
        query=query,
    )
    public_chapters = attach_manifest_citations(public_chapters, citation_manifest)
    chapter_narrative_diagnostics: Dict[str, Any] = {
        "enabled": False,
        "status": "skipped",
        "skipped_reason": "not_run",
    }
    quality_context = _as_dict(analysis_stage_diagnostics)
    if not str(quality_context.get("final_analysis_source") or "").strip():
        has_llm_claim = any(
            "_llm_" in str(item.get("claim_id") or "")
            for item in list(analysis_claim_units or [])
            if isinstance(item, dict)
        )
        if has_llm_claim:
            quality_context = {**quality_context, "final_analysis_source": "llm_evidence_analysis"}
    try:
        public_chapters, chapter_narrative_diagnostics = run_chapter_narrative(
            chapter_packages=public_chapters,
            report_blueprint=report_blueprint,
            llm_config=None,
            quality_context=quality_context,
        )
    except Exception as exc:
        chapter_narrative_diagnostics = {
            "enabled": True,
            "status": "yellow",
            "skipped_reason": "",
            "attempted_count": 0,
            "success_count": 0,
            "fallback_count": len(public_chapters),
            "rejected_count": 0,
            "rejected_reasons": {},
            "failure_reasons": {f"runtime_error:{type(exc).__name__}": 1},
        }
    public_chapters, manifest_section_support = _drop_factual_sections_without_manifest_citations(public_chapters)
    source_claim_support = _merge_source_claim_support_diagnostics(source_claim_support, manifest_section_support)
    before_chapter_count = len(public_chapters)
    public_chapters = [chapter for chapter in public_chapters if _chapter_has_public_body(chapter)]
    omitted_empty_chapters = before_chapter_count - len(public_chapters)
    if omitted_empty_chapters:
        source_claim_support["empty_chapter_omitted_after_source_gate_count"] = (
            int(source_claim_support.get("empty_chapter_omitted_after_source_gate_count") or 0)
            + omitted_empty_chapters
        )
    analysis_transfer = _analysis_transfer_diagnostics(
        claim_units=[
            item
            for item in list(analysis_claim_units or manifest_claim_units or [])
            if isinstance(item, dict)
        ],
        public_chapters=public_chapters,
    )
    source_registry = manifest_appendix_sources(citation_manifest)

    shell = _as_dict(report_blueprint.get("report_shell"))
    front_blocks = _as_block_list(shell.get("front_blocks"), ["executive_summary", "key_data"])
    back_blocks = _as_block_list(shell.get("back_blocks"), ["strategic_options", "risk_triggers"])
    skipped_public_global_blocks: List[str] = []
    front_blocks_for_render: List[str] = []
    for block_key in front_blocks:
        if _public_global_block_allowed(block_key, report_blueprint):
            front_blocks_for_render.append(block_key)
        else:
            skipped_public_global_blocks.append(str(block_key or ""))
    back_blocks_for_render: List[str] = []
    for block_key in back_blocks:
        if _public_global_block_allowed(block_key, report_blueprint):
            back_blocks_for_render.append(block_key)
        else:
            skipped_public_global_blocks.append(str(block_key or ""))
    front_blocks = front_blocks_for_render
    back_blocks = back_blocks_for_render
    summary_title_key = next(
        (block for block in front_blocks if block in SUMMARY_BLOCKS and block not in {"executive_summary", "key_judgments"}),
        front_blocks[0] if front_blocks else "executive_summary",
    )
    parts = [render_cover(query, report_blueprint)]
    rendered_groups: set[str] = set()
    source_appendix_enabled = _source_appendix_enabled(report_blueprint)
    front_section_titles: List[str] = []
    back_section_titles: List[str] = []
    for block_key in front_blocks:
        rendered = _render_global_block(
            block_key,
            title_override=GLOBAL_BLOCK_TITLES.get(summary_title_key, "") if block_key in SUMMARY_BLOCKS else "",
            decision_package=decision_package,
            risk_package=risk_package,
            appendix_package=appendix_package,
            source_registry=source_registry,
            table_packages=table_packages,
            rendered_groups=rendered_groups,
        )
        if rendered:
            parts.append(rendered)
            front_section_titles.append(GLOBAL_BLOCK_TITLES.get(summary_title_key if block_key in SUMMARY_BLOCKS else block_key, block_key))
    for index, chapter in enumerate(public_chapters, start=1):
        parts.append(
            render_chapter_package(
                chapter,
                index,
                previous_chapter=public_chapters[index - 2] if index > 1 else None,
                next_chapter=public_chapters[index] if index < len(public_chapters) else None,
            )
        )
    appendix_requested = False
    for block_key in back_blocks:
        if str(block_key or "").strip() == "appendix":
            appendix_requested = True
            continue
        rendered = _render_global_block(
            block_key,
            title_override="",
            decision_package=decision_package,
            risk_package=risk_package,
            appendix_package=appendix_package,
            source_registry=source_registry,
            table_packages=table_packages,
            rendered_groups=rendered_groups,
        )
        if rendered:
            parts.append(rendered)
            back_section_titles.append(GLOBAL_BLOCK_TITLES.get(block_key, block_key))
    body_markdown = "\n\n".join(part for part in parts if str(part or "").strip())
    body_markdown, source_registry, final_citation_audit = finalize_markdown_citations(
        body_markdown,
        citation_manifest,
        source_registry,
    )
    parts = [body_markdown]
    citation_appendix_required = bool(CITATION_RE.search(body_markdown))
    if (appendix_requested and source_appendix_enabled) or citation_appendix_required:
        rendered = _render_appendix_block(
            title=GLOBAL_BLOCK_TITLES.get("appendix", "appendix"),
            appendix_package=appendix_package,
            source_registry=source_registry,
            rendered_groups=rendered_groups,
        )
        if rendered:
            parts.append(rendered)
            back_section_titles.append(GLOBAL_BLOCK_TITLES.get("appendix", "appendix"))
    markdown = "\n\n".join(part for part in parts if str(part or "").strip())
    markdown = strip_internal_layout_language(markdown)
    markdown = strip_body_qa_leaks(markdown)
    markdown, residual_headline_dropped_count = _drop_residual_headline_lines(markdown)
    markdown, metric_sentence_rewritten_count = _rewrite_bare_metric_lines(markdown)
    markdown = normalize_markdown_spacing(markdown)
    naturalness_before = public_text_artifact_counts(markdown)
    public_narrative_before = public_narrative_leak_audit(markdown)
    markdown = sanitize_public_markdown(markdown)
    public_narrative_after_sanitize = public_narrative_leak_audit(markdown)
    markdown = _renumber_public_chapter_headings(markdown)
    markdown = normalize_markdown_spacing(markdown)
    preliminary_final_citation_audit = dict(final_citation_audit)
    markdown, source_registry, final_citation_audit = _rewrite_final_markdown_with_reconciled_appendix(
        markdown,
        citation_manifest=citation_manifest,
        source_registry=source_registry,
        appendix_package=appendix_package,
    )
    if preliminary_final_citation_audit:
        earlier_removed = _as_list(preliminary_final_citation_audit.get("final_unresolved_citation_refs"))
        if earlier_removed:
            final_citation_audit["final_unresolved_citation_refs"] = _dedupe_strings(
                [*earlier_removed, *_as_list(final_citation_audit.get("final_unresolved_citation_refs"))],
                limit=50,
            )
            final_citation_audit["final_unresolved_citation_removed_count"] = len(
                final_citation_audit["final_unresolved_citation_refs"]
            )
    markdown, final_public_narrative_gate = apply_public_narrative_gate(markdown)
    markdown = normalize_markdown_spacing(markdown)
    audit_before_final_gate = dict(final_citation_audit)
    markdown, source_registry, final_citation_audit = _rewrite_final_markdown_with_reconciled_appendix(
        markdown,
        citation_manifest=citation_manifest,
        source_registry=source_registry,
        appendix_package=appendix_package,
    )
    if audit_before_final_gate:
        earlier_removed = _as_list(audit_before_final_gate.get("final_unresolved_citation_refs"))
        if earlier_removed:
            final_citation_audit["final_unresolved_citation_refs"] = _dedupe_strings(
                [*earlier_removed, *_as_list(final_citation_audit.get("final_unresolved_citation_refs"))],
                limit=50,
            )
            final_citation_audit["final_unresolved_citation_removed_count"] = len(
                final_citation_audit["final_unresolved_citation_refs"]
            )
    markdown = normalize_markdown_spacing(markdown)
    public_narrative_after = public_narrative_leak_audit(markdown)
    public_narrative_gate = {
        "skipped_global_blocks": [item for item in skipped_public_global_blocks if item],
        "skipped_global_block_count": len([item for item in skipped_public_global_blocks if item]),
        "public_narrative_leak_input_count": public_narrative_before.get("blocker_count", 0),
        "public_narrative_leak_after_sanitize_count": public_narrative_after_sanitize.get("blocker_count", 0),
        "public_narrative_leak_remaining_count": public_narrative_after.get("blocker_count", 0),
        "blocker_count": public_narrative_after.get("blocker_count", 0),
        "public_narrative_leak_removed_count": max(
            0,
            int(public_narrative_before.get("blocker_count", 0) or 0)
            - int(public_narrative_after.get("blocker_count", 0) or 0),
        ),
        "public_narrative_leak_reason_counts": public_narrative_before.get("reason_counts", {}),
        "public_narrative_leak_examples": public_narrative_before.get("examples", []),
        "public_narrative_leak_remaining_examples": public_narrative_after.get("examples", []),
        "final_gate": final_public_narrative_gate,
    }
    naturalness_after = public_text_artifact_counts(markdown)
    warnings = collect_format_warnings(markdown)
    return {
        "agent": AGENT_NAME,
        "report_markdown": markdown,
        "sections": [
            *front_section_titles,
            *[str(chapter.get("chapter_title") or "") for chapter in public_chapters],
            *back_section_titles,
        ],
        "source_count": len(source_registry),
        "source_registry": list(source_registry),
        "citation_manifest": citation_manifest,
        "final_citation_audit": final_citation_audit,
        "source_claim_support": source_claim_support,
        "chapter_narrative": chapter_narrative_diagnostics,
        "analysis_transfer": analysis_transfer,
        "ref_lineage_diagnostics": ref_lineage_diagnostics,
        "public_narrative_leak_audit": public_narrative_gate,
        "excluded_untraceable_source_count": len(excluded_sources),
        "excluded_untraceable_sources": [
            {
                "ref": source.get("ref"),
                "title": source.get("title") or source.get("source_title"),
                "reason": "title_only_or_untraceable",
            }
            for source in excluded_sources[:20]
        ],
        "estimated_chars": len(markdown),
        "format_warnings": warnings,
        "naturalness_cleanup": {
            "residual_headline_dropped_count": residual_headline_dropped_count,
            "metric_sentence_rewritten_count": metric_sentence_rewritten_count,
            **naturalness_before,
            "residual_ocr_artifact_count": naturalness_after.get("ocr_artifact_normalized_count", 0),
            "residual_traditional_chinese_count": naturalness_after.get("traditional_chinese_normalized_count", 0),
            "residual_empty_parens_count": naturalness_after.get("empty_parens_removed_count", 0),
            "residual_truncated_punctuation_count": naturalness_after.get("truncated_punctuation_cleaned_count", 0),
        },
    }
