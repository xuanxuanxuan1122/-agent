from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def _strip_code_fence(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text


def _json_start(text: str) -> int:
    positions = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    return min(positions) if positions else -1


def _close_truncated_json(text: str) -> str:
    stack: List[str] = []
    in_string = False
    escape = False
    last_safe_index = -1
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
                last_safe_index = index
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()
            last_safe_index = index
        elif char in ",:" or not char.isspace():
            last_safe_index = index
    if in_string and last_safe_index >= 0:
        text = text[: last_safe_index + 1]
        while text.rstrip().endswith((",", ":")):
            text = text.rstrip()[:-1]
    else:
        text = text.rstrip()
        while text.endswith((",", ":")):
            text = text[:-1].rstrip()
    return text + "".join(reversed(stack))


def _try_load(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def salvage_json_payload(raw_text: str) -> Dict[str, Any]:
    text = _strip_code_fence(raw_text)
    start = _json_start(text)
    if start < 0:
        return {}
    text = text[start:].strip()
    loaded = _try_load(text)
    if isinstance(loaded, dict):
        return loaded
    closed = _close_truncated_json(text)
    loaded = _try_load(closed)
    return loaded if isinstance(loaded, dict) else {}


def salvage_chapter_json(raw_text: str, *, chapter_id: str = "", chapter_title: str = "") -> Dict[str, Any]:
    payload = salvage_json_payload(raw_text)
    if not payload:
        return {}
    if "chapter_synthesis" in payload:
        return payload
    claim_units = payload.get("claim_units")
    if isinstance(claim_units, list):
        return {
            "chapter_synthesis": [
                {
                    "chapter_id": payload.get("chapter_id") or chapter_id,
                    "chapter_title": payload.get("chapter_title") or chapter_title,
                    "claim_units": claim_units,
                    "analysis_limits": payload.get("analysis_limits") or [],
                }
            ],
            "analysis_limits": payload.get("analysis_limits") or [],
        }
    return payload
