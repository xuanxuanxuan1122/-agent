from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _dedupe_refs(values: Sequence[Any], *, limit: int = 8) -> List[str]:
    refs: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        refs.append(text)
        if len(refs) >= limit:
            break
    return refs


def pick_refs(item: Dict[str, Any], *, limit: int = 6) -> List[str]:
    """Return stable citation refs from any evidence-shaped payload."""
    refs: List[Any] = []
    for key in ("source_refs", "evidence_refs", "supporting_evidence", "refs"):
        refs.extend(_as_list(item.get(key)))
    for key in ("source_ref", "citation_ref", "ref"):
        value = str(item.get(key) or "").strip()
        if value:
            refs.append(value)
    source_id = str(item.get("source_id") or "").strip()
    if re.fullmatch(r"\d{1,5}", source_id):
        refs.append(f"[{source_id}]")
    return _dedupe_refs(refs, limit=limit)


def renumber_sources_by_first_citation(
    markdown: str,
    source_registry: Sequence[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Make source ids follow first citation order without inventing refs."""
    sources = [dict(item) for item in list(source_registry or []) if isinstance(item, dict)]
    if not markdown or not sources:
        return markdown, sources
    by_ref = {str(source.get("ref") or "").strip(): source for source in sources if str(source.get("ref") or "").strip()}
    if not by_ref:
        return markdown, sources

    seen_refs: List[str] = []
    for match in re.finditer(r"\[(\d{1,5})\]", markdown):
        ref = f"[{match.group(1)}]"
        if ref in by_ref and ref not in seen_refs:
            seen_refs.append(ref)
    ordered_refs = seen_refs + [
        str(source.get("ref") or "").strip()
        for source in sources
        if str(source.get("ref") or "").strip() and str(source.get("ref") or "").strip() not in seen_refs
    ]
    mapping = {old_ref: f"[{index}]" for index, old_ref in enumerate(ordered_refs, start=1)}

    def replace_ref(match: re.Match[str]) -> str:
        ref = f"[{match.group(1)}]"
        return mapping.get(ref, "")

    rewritten_markdown = re.sub(r"\[(\d{1,5})\]", replace_ref, markdown)
    ordered_sources: List[Dict[str, Any]] = []
    for index, old_ref in enumerate(ordered_refs, start=1):
        source = dict(by_ref.get(old_ref) or {})
        if not source:
            continue
        source["ref"] = f"[{index}]"
        source["source_id"] = f"SRC-{index:03d}"
        ordered_sources.append(source)
    return rewritten_markdown, ordered_sources
