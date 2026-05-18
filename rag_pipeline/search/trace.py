from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .models import SearchTrace


def trim_candidate_payloads(items: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    trimmed: List[Dict[str, Any]] = []
    for item in list(items)[: max(1, int(top_k))]:
        entry = dict(item)
        text = str(entry.get("text") or "").strip()
        if text:
            entry["text"] = text[:600]
        preview = str(entry.get("text_preview") or "").strip()
        if preview:
            entry["text_preview"] = preview[:400]
        trimmed.append(entry)
    return trimmed


class TraceRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        trace_dir: Path | str,
        session_id: str,
        turn_id: str,
        original_query: str,
        standalone_query: str,
    ):
        self.enabled = bool(enabled)
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace = SearchTrace(
            session_id=session_id,
            turn_id=turn_id,
            original_query=original_query,
            standalone_query=standalone_query,
        )

    def set_memory(self, memory_payload: Dict[str, Any]) -> None:
        if self.enabled:
            self.trace.memory = dict(memory_payload or {})

    def set_plan(self, plan_payload: Dict[str, Any]) -> None:
        if self.enabled:
            self.trace.plan = dict(plan_payload or {})

    def add_llm_call(self, llm_payload: Optional[Dict[str, Any]]) -> None:
        if self.enabled and llm_payload:
            self.trace.llm_calls.append(dict(llm_payload))

    def add_hop(
        self,
        *,
        hop_index: int,
        query: str,
        retrieved: Sequence[Dict[str, Any]],
        reranked: Sequence[Dict[str, Any]],
        evidence: Sequence[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> None:
        if not self.enabled:
            return
        self.trace.hops.append(
            {
                "hop_index": int(hop_index),
                "query": str(query or "").strip(),
                "retrieved": trim_candidate_payloads(retrieved, top_k=top_k),
                "reranked": trim_candidate_payloads(reranked, top_k=top_k),
                "evidence": list(evidence)[: max(1, top_k)],
                "metadata": dict(metadata or {}),
            }
        )

    def set_evidence(self, evidence_payload: Sequence[Dict[str, Any]]) -> None:
        if self.enabled:
            self.trace.evidence = list(evidence_payload)

    def set_answer(self, answer_payload: Dict[str, Any]) -> None:
        if self.enabled:
            self.trace.answer = dict(answer_payload or {})

    def set_timings(self, timings: Dict[str, float]) -> None:
        if self.enabled:
            self.trace.timings = {key: float(value or 0.0) for key, value in (timings or {}).items()}

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        if self.enabled:
            self.trace.metadata = dict(metadata or {})

    def write(self) -> str:
        if not self.enabled:
            return ""
        session_dir = self.trace_dir / (self.trace.session_id or "standalone")
        session_dir.mkdir(parents=True, exist_ok=True)
        output_path = session_dir / f"{self.trace.turn_id}.json"
        output_path.write_text(json.dumps(self.trace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(output_path)
