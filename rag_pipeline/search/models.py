from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SearchCandidate:
    id: str
    chunk_uid: str
    chunk_level: str
    parent_chunk_uid: str
    source_name: str
    semantic_score: float
    doc_title: str
    source_file: str
    section_title: str
    header_path: List[str]
    chunk_type: str
    quality_score: float
    text_preview: str
    text: str
    payload: Dict[str, Any]
    matched_queries: List[str]
    score_breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "chunk_uid": self.chunk_uid,
            "chunk_level": self.chunk_level,
            "parent_chunk_uid": self.parent_chunk_uid,
            "source_name": self.source_name,
            "semantic_score": self.semantic_score,
            "doc_title": self.doc_title,
            "source_file": self.source_file,
            "section_title": self.section_title,
            "header_path": self.header_path,
            "chunk_type": self.chunk_type,
            "quality_score": self.quality_score,
            "section_kind": self.payload.get("section_kind", ""),
            "page_no": self.payload.get("page_no", 0),
            "page_label": self.payload.get("page_label", ""),
            "knowledge_unit_type": self.payload.get("knowledge_unit_type", ""),
            "info_density": self.payload.get("info_density", 0.0),
            "noise_score": self.payload.get("noise_score", 0.0),
            "summary_consistency_score": self.payload.get("summary_consistency_score", 0.0),
            "text_preview": self.text_preview,
            "text": self.text,
            "matched_queries": self.matched_queries,
            "score_breakdown": self.score_breakdown,
        }


@dataclass
class EvidenceItem:
    rank: int
    chunk_uid: str
    source_file: str
    doc_title: str
    section_title: str
    chunk_level: str
    group: str
    quote: str
    evidence_score: float
    body_support_score: float
    answerability_score: float
    final_score: float
    citation: Dict[str, str]
    tier: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_uid": self.chunk_uid,
            "source_file": self.source_file,
            "doc_title": self.doc_title,
            "section_title": self.section_title,
            "chunk_level": self.chunk_level,
            "group": self.group,
            "quote": self.quote,
            "evidence_score": self.evidence_score,
            "body_support_score": self.body_support_score,
            "answerability_score": self.answerability_score,
            "final_score": self.final_score,
            "citation": self.citation,
            "tier": self.tier,
        }


@dataclass
class AnswerSynthesis:
    status: str
    confidence: float
    answer: str
    claims: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    refusal_reason: str = ""
    citations: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    followups: List[str] = field(default_factory=list)
    grounding_mode: str = "extractive"
    llm_model: str = ""
    review_status: str = ""
    review_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "answer": self.answer,
            "claims": self.claims,
            "conflicts": self.conflicts,
            "refusal_reason": self.refusal_reason,
            "citations": self.citations,
            "gaps": self.gaps,
            "followups": self.followups,
            "grounding_mode": self.grounding_mode,
            "llm_model": self.llm_model,
            "review_status": self.review_status,
            "review_issues": self.review_issues,
        }


@dataclass
class QueryPlan:
    original_query: str
    intent: str
    task_type: str
    normalized_query: str
    theme_terms: List[str] = field(default_factory=list)
    entity_terms: List[str] = field(default_factory=list)
    constraint_terms: List[str] = field(default_factory=list)
    time_terms: List[str] = field(default_factory=list)
    evidence_focus: List[str] = field(default_factory=list)
    filter_hints: Dict[str, List[str]] = field(default_factory=dict)
    sub_queries: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    needs_multi_hop: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "intent": self.intent,
            "task_type": self.task_type,
            "normalized_query": self.normalized_query,
            "theme_terms": self.theme_terms,
            "entity_terms": self.entity_terms,
            "constraint_terms": self.constraint_terms,
            "time_terms": self.time_terms,
            "evidence_focus": self.evidence_focus,
            "filter_hints": self.filter_hints,
            "sub_queries": self.sub_queries,
            "search_queries": self.search_queries,
            "notes": self.notes,
            "needs_multi_hop": self.needs_multi_hop,
        }


@dataclass
class Turn:
    turn_id: str
    user_query: str
    standalone_query: str = ""
    intent: str = ""
    task_type: str = ""
    answer_status: str = ""
    answer: str = ""
    summary: str = ""
    retrieved_chunk_uids: List[str] = field(default_factory=list)
    evidence_chunk_uids: List[str] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    trace_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_query": self.user_query,
            "standalone_query": self.standalone_query,
            "intent": self.intent,
            "task_type": self.task_type,
            "answer_status": self.answer_status,
            "answer": self.answer,
            "summary": self.summary,
            "retrieved_chunk_uids": self.retrieved_chunk_uids,
            "evidence_chunk_uids": self.evidence_chunk_uids,
            "plan": self.plan,
            "timings": self.timings,
            "trace_file": self.trace_file,
            "metadata": self.metadata,
        }


@dataclass
class AgentState:
    session_id: str
    turn_id: str
    user_query: str
    standalone_query: str = ""
    memory_summary: str = ""
    recent_turns: List[Dict[str, Any]] = field(default_factory=list)
    reused_chunk_uids: List[str] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    reflections: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "user_query": self.user_query,
            "standalone_query": self.standalone_query,
            "memory_summary": self.memory_summary,
            "recent_turns": self.recent_turns,
            "reused_chunk_uids": self.reused_chunk_uids,
            "plan": self.plan,
            "reflections": self.reflections,
            "metadata": self.metadata,
        }


@dataclass
class AgentResponse:
    session_id: str
    turn_id: str
    original_query: str
    standalone_query: str
    plan: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    answer: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    trace_file: str = ""
    memory_updated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "original_query": self.original_query,
            "standalone_query": self.standalone_query,
            "plan": self.plan,
            "results": self.results,
            "evidence": self.evidence,
            "answer": self.answer,
            "timings": self.timings,
            "trace_file": self.trace_file,
            "memory_updated": self.memory_updated,
            "metadata": self.metadata,
        }


@dataclass
class SearchTrace:
    session_id: str
    turn_id: str
    original_query: str
    standalone_query: str
    plan: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    hops: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    answer: Dict[str, Any] = field(default_factory=dict)
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "original_query": self.original_query,
            "standalone_query": self.standalone_query,
            "plan": self.plan,
            "memory": self.memory,
            "hops": self.hops,
            "evidence": self.evidence,
            "answer": self.answer,
            "llm_calls": self.llm_calls,
            "timings": self.timings,
            "metadata": self.metadata,
        }
