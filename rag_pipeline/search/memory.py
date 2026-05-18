from __future__ import annotations

import json
import re
import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from ..config.search_config import DEFAULT_EXTERNAL_API_TRUST_ENV, DEFAULT_LLM_DISABLE_THINKING
from .models import Turn


_PRONOUN_RE = re.compile(r"(?:^|\b)(it|they|them|that|this|he|she|its|their)\b|[它那个这该其他们她们该公司该行业该技术]", re.I)


def generate_turn_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _normalize_bool_flag(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_url_opener(*, trust_env: bool) -> urllib.request.OpenerDirector:
    if trust_env:
        return urllib.request.build_opener()
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def normalize_llm_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = dict(config or {})
    return {
        "provider": str(config.get("provider") or "openai_compatible").strip().lower() or "openai_compatible",
        "url": str(config.get("url") or "").strip(),
        "api_key": str(config.get("api_key") or "").strip(),
        "model": str(config.get("model") or "").strip(),
        "timeout": float(config.get("timeout") or 30.0),
        "trust_env": _normalize_bool_flag(config.get("trust_env"), DEFAULT_EXTERNAL_API_TRUST_ENV),
        "disable_thinking": _normalize_bool_flag(config.get("disable_thinking"), DEFAULT_LLM_DISABLE_THINKING),
    }


def llm_config_is_ready(config: Optional[Dict[str, Any]]) -> bool:
    normalized = normalize_llm_config(config)
    return bool(normalized["url"] and normalized["api_key"] and normalized["model"])


def normalize_openai_compatible_chat_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return cleaned


def call_openai_compatible_json(
    *,
    config: Dict[str, Any],
    system_prompt: str,
    user_payload: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = normalize_llm_config(config)
    if normalized["provider"] != "openai_compatible":
        raise RuntimeError(f"Unsupported LLM provider: {normalized['provider']}")
    if not llm_config_is_ready(normalized):
        raise RuntimeError("LLM config is incomplete.")

    payload = {
        "model": normalized["model"],
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }
    if normalized.get("disable_thinking"):
        payload["enable_thinking"] = False
    request = urllib.request.Request(
        normalize_openai_compatible_chat_url(normalized["url"]),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {normalized['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = _build_url_opener(trust_env=normalized["trust_env"])
    try:
        with opener.open(request, timeout=normalized["timeout"]) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response did not include choices.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        content = "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    content = str(content or "").strip()
    if not content:
        raise RuntimeError("LLM response content is empty.")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM response is not valid JSON: {content[:400]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("LLM response JSON root must be an object.")
    return {
        "payload": payload,
        "usage": data.get("usage", {}),
        "raw_response": data,
        "request_payload": user_payload,
    }


class ConversationMemory:
    def __init__(self, store_dir: Path | str, max_turns: int = 10, summary_trigger: int = 6):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.max_turns = max(1, int(max_turns))
        self.summary_trigger = max(2, int(summary_trigger))

    def _session_path(self, session_id: str) -> Path:
        raw = str(session_id or "").strip()
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw) or "default"
        digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12] if raw else "empty"
        return self.store_dir / f"{safe[:80]}-{digest}.json"

    def load_session(self, session_id: str) -> Dict[str, Any]:
        if not session_id:
            return {"session_id": "", "summary": "", "turns": []}
        path = self._session_path(session_id)
        if not path.exists():
            return {"session_id": session_id, "summary": "", "turns": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"session_id": session_id, "summary": "", "turns": []}
        turns = data.get("turns", [])
        if not isinstance(turns, list):
            turns = []
        return {
            "session_id": session_id,
            "summary": str(data.get("summary") or "").strip(),
            "turns": turns,
        }

    def save_session(self, session_id: str, summary: str, turns: Iterable[Dict[str, Any]]) -> None:
        if not session_id:
            return
        payload = {
            "session_id": session_id,
            "summary": str(summary or "").strip(),
            "turns": list(turns),
        }
        self._session_path(session_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def get_recent_turns(self, session_id: str, limit: int = 4) -> List[Dict[str, Any]]:
        session = self.load_session(session_id)
        return list(session.get("turns", []))[-max(1, int(limit)) :]

    def get_recent_retrieved_chunk_uids(self, session_id: str, limit: int = 1) -> List[str]:
        turns = self.get_recent_turns(session_id, limit=limit)
        values: List[str] = []
        seen = set()
        for turn in reversed(turns):
            for field in ("evidence_chunk_uids", "retrieved_chunk_uids"):
                for uid in turn.get(field, []) or []:
                    cleaned = str(uid or "").strip()
                    if cleaned and cleaned not in seen:
                        seen.add(cleaned)
                        values.append(cleaned)
        return values

    def append_turn(
        self,
        session_id: str,
        turn: Turn,
        *,
        llm_config: Optional[Dict[str, Any]] = None,
        llm_trace_collector: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not session_id:
            return
        session = self.load_session(session_id)
        turns = list(session.get("turns", []))
        turns.append(turn.to_dict())
        summary = str(session.get("summary") or "").strip()

        if len(turns) > self.max_turns:
            overflow = turns[:-self.max_turns]
            turns = turns[-self.max_turns :]
            summary = summarize_archived_turns(
                current_summary=summary,
                archived_turns=overflow,
                llm_config=llm_config,
                llm_trace_collector=llm_trace_collector,
            )

        self.save_session(session_id, summary=summary, turns=turns)


def build_turn_summary(query: str, answer: str) -> str:
    answer = " ".join(str(answer or "").split())
    if not answer:
        return str(query or "").strip()[:180]
    return f"Q: {str(query or '').strip()} | A: {answer[:220]}"


def summarize_archived_turns(
    *,
    current_summary: str,
    archived_turns: List[Dict[str, Any]],
    llm_config: Optional[Dict[str, Any]] = None,
    llm_trace_collector: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if not archived_turns:
        return current_summary

    compact_items = []
    for turn in archived_turns:
        compact_items.append(
            {
                "query": str(turn.get("user_query") or "").strip(),
                "standalone_query": str(turn.get("standalone_query") or "").strip(),
                "answer_status": str(turn.get("answer_status") or "").strip(),
                "summary": str(turn.get("summary") or "").strip(),
            }
        )

    if llm_config_is_ready(llm_config):
        system_prompt = (
            "你负责总结企业检索问答中的历史对话轮次。"
            "只返回 JSON，且只能包含一个键：summary。"
            "请把归档轮次压缩成简洁、事实性的会话记忆，保留实体、约束条件和仍未解决的追问。"
        )
        user_payload = {
            "current_summary": current_summary,
            "archived_turns": compact_items,
        }
        try:
            response = call_openai_compatible_json(config=llm_config or {}, system_prompt=system_prompt, user_payload=user_payload)
            payload = response.get("payload", {})
            summary = str(payload.get("summary") or "").strip()
            if summary:
                if llm_trace_collector is not None:
                    llm_trace_collector.append(
                        {
                            "type": "memory_summary",
                            "request": user_payload,
                            "response": payload,
                            "usage": response.get("usage", {}),
                        }
                    )
                return summary
        except Exception as exc:
            if llm_trace_collector is not None:
                llm_trace_collector.append(
                    {
                        "type": "memory_summary",
                        "request": user_payload,
                        "error": str(exc),
                    }
                )

    lines = [part for part in [current_summary] if part]
    lines.extend(item["summary"] or item["query"] for item in compact_items if item["summary"] or item["query"])
    merged = " | ".join(lines)
    return merged[:800]


def build_contextualization_result(
    *,
    query: str,
    standalone_query: str,
    memory_summary: str,
    recent_turns: List[Dict[str, Any]],
    reused_chunk_uids: List[str],
    source: str,
    note: str = "",
    llm_call: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "original_query": str(query or "").strip(),
        "standalone_query": str(standalone_query or query or "").strip(),
        "memory_summary": memory_summary,
        "recent_turns": recent_turns,
        "reused_chunk_uids": reused_chunk_uids,
        "source": source,
        "note": note,
        "llm_call": llm_call or {},
    }


def contextualize_query(
    *,
    query: str,
    session_id: str,
    memory: ConversationMemory,
    llm_config: Optional[Dict[str, Any]],
    history_turns: int = 4,
) -> Dict[str, Any]:
    query = str(query or "").strip()
    if not session_id:
        return build_contextualization_result(
            query=query,
            standalone_query=query,
            memory_summary="",
            recent_turns=[],
            reused_chunk_uids=[],
            source="disabled",
            note="no_session_id",
        )

    session = memory.load_session(session_id)
    recent_turns = list(session.get("turns", []))[-max(1, int(history_turns)) :]
    memory_summary = str(session.get("summary") or "").strip()
    reused_chunk_uids = memory.get_recent_retrieved_chunk_uids(session_id, limit=1)

    if not recent_turns:
        return build_contextualization_result(
            query=query,
            standalone_query=query,
            memory_summary=memory_summary,
            recent_turns=[],
            reused_chunk_uids=[],
            source="cold_start",
            note="no_prior_turns",
        )

    heuristic_followup = bool(_PRONOUN_RE.search(query))
    if llm_config_is_ready(llm_config):
        system_prompt = (
            "你负责把企业检索问答中的追问改写成可独立检索的查询。"
            "不要回答问题，只消解指代并保留原有约束。"
            "只返回 JSON，包含这些键：standalone_query、reuse_last_turn、note。"
        )
        recent_payload = [
            {
                "user_query": str(turn.get("user_query") or "").strip(),
                "standalone_query": str(turn.get("standalone_query") or "").strip(),
                "summary": str(turn.get("summary") or "").strip(),
                "answer_status": str(turn.get("answer_status") or "").strip(),
            }
            for turn in recent_turns
        ]
        user_payload = {
            "current_query": query,
            "memory_summary": memory_summary,
            "recent_turns": recent_payload,
        }
        try:
            response = call_openai_compatible_json(config=llm_config or {}, system_prompt=system_prompt, user_payload=user_payload)
            payload = response.get("payload", {})
            standalone_query = str(payload.get("standalone_query") or query).strip() or query
            reuse_last_turn = bool(payload.get("reuse_last_turn", False))
            if reuse_last_turn:
                reused_chunk_uids = memory.get_recent_retrieved_chunk_uids(session_id, limit=1)
            return build_contextualization_result(
                query=query,
                standalone_query=standalone_query,
                memory_summary=memory_summary,
                recent_turns=recent_payload,
                reused_chunk_uids=reused_chunk_uids,
                source="llm",
                note=str(payload.get("note") or "").strip(),
                llm_call={
                    "type": "contextualize_query",
                    "request": user_payload,
                    "response": payload,
                    "usage": response.get("usage", {}),
                },
            )
        except Exception as exc:
            pass

    if heuristic_followup and recent_turns:
        last_turn = recent_turns[-1]
        anchor = str(last_turn.get("standalone_query") or last_turn.get("user_query") or "").strip()
        standalone_query = f"{anchor} ; follow-up: {query}".strip(" ;")
        return build_contextualization_result(
            query=query,
            standalone_query=standalone_query,
            memory_summary=memory_summary,
            recent_turns=recent_turns,
            reused_chunk_uids=reused_chunk_uids,
            source="heuristic",
            note="pronoun_follow_up",
        )

    return build_contextualization_result(
        query=query,
        standalone_query=query,
        memory_summary=memory_summary,
        recent_turns=recent_turns,
        reused_chunk_uids=[],
        source="passthrough",
        note="history_present_but_no_rewrite_needed",
    )
