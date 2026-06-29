# Evidence Curator Analysis Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore evidence-to-analysis recall by replacing over-strict source/metric gates with dirty-data quarantine plus model-driven evidence curation before claim generation.

**Architecture:** Keep hard blocking only for dirty or non-public data. Insert an `Evidence Curator` stage between `evidence_merge` and `analysis_agent` that cleans, condenses, dedupes, labels, and weights evidence for analysis. Feed analysis with curated notes and diverse evidence bundles so useful C/D/media/professional-site signals can become bounded directional claims instead of being discarded.

**Tech Stack:** Python, pytest, existing RAG pipeline agents under `current_rag_pipeline/rag_pipeline/agents`, existing observability probes, existing LLM routing helpers.

---

## File Structure

- Create: `current_rag_pipeline/rag_pipeline/agents/evidence_curator_agent.py`
  - Owns model-assisted evidence cleaning and curation.
  - Exposes pure helpers plus optional LLM call wrapper.

- Create: `current_rag_pipeline/rag_pipeline/contracts/evidence_dirty_gate.py`
  - Owns the only hard-blocking dirty-data rules.
  - Returns deterministic `DirtyGateResult` dictionaries.

- Modify: `current_rag_pipeline/rag_pipeline/agents/evidence_merger.py`
  - Stop treating source level and metric incompleteness as hard analysis exclusion.
  - Attach dirty-gate diagnostics and pass non-dirty evidence forward.

- Modify: `current_rag_pipeline/rag_pipeline/agents/readpage_fact_extractor_agent.py`
  - Do not reject fact cards only because `source_level_below_required`.
  - Downgrade to `directional_signal` with limitation metadata.

- Modify: `current_rag_pipeline/rag_pipeline/agents/analysis_agent.py`
  - Build analysis input from curated evidence when available.
  - Add a two-pass analysis contract: evidence inventory first, claim units second.
  - Increase recall through diverse sampling without feeding raw noisy pages.

- Modify: `current_rag_pipeline/rag_pipeline/flows/report/full_report.py`
  - Wire the curator stage after `evidence_package` creation and before structured analysis.
  - Add env flags and stage snapshots/probes.

- Modify: `current_rag_pipeline/rag_pipeline/observability/evidence_claim_conversion.py`
  - Track curated evidence counts and curated-to-claim conversion.

- Test: `current_rag_pipeline/tests/test_evidence_dirty_gate.py`
- Test: `current_rag_pipeline/tests/test_evidence_curator_agent.py`
- Test: `current_rag_pipeline/tests/test_readpage_fact_extractor_agent.py`
- Test: `current_rag_pipeline/tests/test_analysis_main_chain_contract.py`
- Test: `current_rag_pipeline/tests/test_evidence_claim_conversion.py`

---

## Task 1: Add Dirty Data Quarantine Contract

**Files:**
- Create: `current_rag_pipeline/rag_pipeline/contracts/evidence_dirty_gate.py`
- Test: `current_rag_pipeline/tests/test_evidence_dirty_gate.py`

- [ ] **Step 1: Write failing tests for dirty-only blocking**

Create `current_rag_pipeline/tests/test_evidence_dirty_gate.py`:

```python
from rag_pipeline.contracts.evidence_dirty_gate import evaluate_dirty_gate


def test_dirty_gate_blocks_login_and_navigation_pages():
    item = {
        "evidence_id": "EV-login",
        "fact": "Login Sign in Cookie Privacy Policy Please enable JavaScript",
        "source": {"url": "https://example.com/login", "title": "Login"},
    }

    result = evaluate_dirty_gate(item)

    assert result["status"] == "blocked"
    assert "page_shell_or_login" in result["reasons"]
    assert result["public_text_allowed"] is False


def test_dirty_gate_allows_traceable_low_tier_industry_signal():
    item = {
        "evidence_id": "EV-signal",
        "fact": "前瞻产业研究院称，2025年全球人形机器人出货约1.7万台，产业进入小批量验证阶段。",
        "source_level": "C",
        "source": {"url": "https://example.com/report", "title": "产业研究报告"},
    }

    result = evaluate_dirty_gate(item)

    assert result["status"] == "allowed"
    assert result["reasons"] == []
    assert result["public_text_allowed"] is True


def test_dirty_gate_does_not_block_incomplete_metric_by_itself():
    item = {
        "evidence_id": "EV-metric",
        "fact": "机构预测中国人形机器人市场规模将在2026年突破200亿元。",
        "metric": "市场规模",
        "value": "200亿元",
        "unit": "",
        "period": "",
        "source": {"url": "https://example.com/market", "title": "市场预测"},
    }

    result = evaluate_dirty_gate(item)

    assert result["status"] == "allowed"
    assert "metric_incomplete" in result["warnings"]
    assert "metric_incomplete" not in result["reasons"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_evidence_dirty_gate.py -q -o addopts=""
```

Expected: FAIL because `rag_pipeline.contracts.evidence_dirty_gate` does not exist.

- [ ] **Step 3: Implement dirty gate**

Create `current_rag_pipeline/rag_pipeline/contracts/evidence_dirty_gate.py`:

```python
from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urlparse


PAGE_SHELL_RE = re.compile(
    r"(login|sign\s*in|cookie|privacy\s*policy|terms\s*of\s*use|"
    r"please\s+enable\s+javascript|checking\s+your\s+browser|request\s+a\s+demo|"
    r"book\s+a\s+demo|跳转|登录|注册|隐私政策|用户协议|无障碍浏览)",
    re.I,
)

INTERNAL_MARKER_RE = re.compile(
    r"\b(EV-\d+|claim_unit|score_gap|diagnostic_only|source_check|"
    r"analysis_ready_exclusion_reason|llm_semantic_judge)\b",
    re.I,
)

PLACEHOLDER_URL_RE = re.compile(r"(example\.(com|org|net)|placeholder|localhost|127\.0\.0\.1)", re.I)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    return " ".join(
        str(value or "")
        for value in (
            item.get("fact"),
            item.get("clean_fact"),
            item.get("content"),
            item.get("distilled_fact"),
            item.get("metric"),
            item.get("value"),
            source.get("title"),
            source.get("url"),
            item.get("source_url"),
        )
    )


def _source_url(item: Dict[str, Any]) -> str:
    source = _as_dict(item.get("source"))
    return str(item.get("source_url") or source.get("url") or source.get("source_url") or "").strip()


def _has_source_ref(item: Dict[str, Any]) -> bool:
    source = _as_dict(item.get("source"))
    return bool(
        _source_url(item)
        or str(item.get("source_ref") or item.get("source_id") or item.get("run_source_id") or "").strip()
        or str(source.get("source_ref") or source.get("document_id") or source.get("page_ref") or "").strip()
    )


def _looks_like_pure_title_or_number(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", compact):
        return True
    if len(compact) < 12 and not re.search(r"[。；，,;:：]", text):
        return True
    return False


def evaluate_dirty_gate(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return dirty-data quarantine status.

    This contract blocks only data that should never reach public analysis.
    Weak source level, single source, and incomplete metrics are warnings, not
    blocking reasons.
    """

    text = _text(item)
    url = _source_url(item)
    reasons: List[str] = []
    warnings: List[str] = []

    if not _has_source_ref(item):
        reasons.append("missing_source_ref")
    if url and PLACEHOLDER_URL_RE.search(url):
        reasons.append("placeholder_or_fake_url")
    if PAGE_SHELL_RE.search(text):
        reasons.append("page_shell_or_login")
    if INTERNAL_MARKER_RE.search(text):
        reasons.append("internal_diagnostic_marker")
    if _looks_like_pure_title_or_number(str(item.get("fact") or item.get("clean_fact") or item.get("distilled_fact") or item.get("content") or "")):
        warnings.append("thin_or_title_like_text")

    metric = str(item.get("metric") or "").strip()
    value = str(item.get("value") or "").strip()
    unit = str(item.get("unit") or "").strip()
    period = str(item.get("period") or "").strip()
    if metric and (not value or not unit or not period):
        warnings.append("metric_incomplete")

    parsed_host = urlparse(url).netloc.lower() if url else ""
    if parsed_host and parsed_host.endswith(".pdf") and len(text) < 80:
        warnings.append("pdf_metadata_only")

    status = "blocked" if reasons else "allowed"
    return {
        "schema_version": "dirty_gate_v1",
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "diagnostic_only": status == "blocked",
        "must_not_render": status == "blocked",
        "public_text_allowed": status != "blocked",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_evidence_dirty_gate.py -q -o addopts=""
```

Expected: PASS.

---

## Task 2: Relax Readpage Extraction Without Letting Dirty Data Through

**Files:**
- Modify: `current_rag_pipeline/rag_pipeline/agents/readpage_fact_extractor_agent.py`
- Test: `current_rag_pipeline/tests/test_readpage_fact_extractor_agent.py`

- [ ] **Step 1: Add failing test for source-level downgrade**

Append to `current_rag_pipeline/tests/test_readpage_fact_extractor_agent.py`:

```python
def test_readpage_fact_card_low_source_level_is_directional_not_rejected():
    from rag_pipeline.agents import readpage_fact_extractor_agent as agent

    card = {
        "distilled_fact": "垂直媒体报道称，2025年人形机器人商业化订单开始增多。",
        "fact_type": "case",
        "source_url": "https://example.org/news/humanoid",
        "source_level": "C",
        "source_verification_status": "readpage_verified",
        "proof_role": "case",
    }
    task = {"required_source_level": ["A", "B"], "proof_role": "case"}

    normalized, rejected = agent._normalize_fact_card(card, index=0, task=task)

    assert rejected == []
    assert normalized is not None
    assert normalized["allowed_use"] == "directional_signal"
    assert normalized["claim_strength_hint"] == "directional"
    assert normalized["source_level_gap"] == {"required": ["A", "B"], "actual": "C"}
```

- [ ] **Step 2: Run the targeted test**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_readpage_fact_extractor_agent.py::test_readpage_fact_card_low_source_level_is_directional_not_rejected -q -o addopts=""
```

Expected: FAIL because low source level currently appends `source_level_below_required` rejection.

- [ ] **Step 3: Modify source-level handling**

In `current_rag_pipeline/rag_pipeline/agents/readpage_fact_extractor_agent.py`, replace the block that rejects `source_level_below_required` with:

```python
    required_source_levels = {
        str(item or "").strip().upper()
        for item in _as_list(task.get("required_source_level") or task.get("min_source_level"))
        if str(item or "").strip()
    }
    if required_source_levels and card["source_level"] not in required_source_levels:
        card["source_level_gap"] = {
            "required": sorted(required_source_levels),
            "actual": card["source_level"],
        }
        card["claim_strength_hint"] = "directional"
        card["allowed_use"] = "directional_signal"
        card["evidence_use_level"] = "directional_signal"
        card["limitation_boundary"] = _as_list(card.get("limitation_boundary")) + [
            "来源等级低于任务偏好，只能作为方向性信号，不能单独支撑强结论。"
        ]
```

Keep dirty text, fake URL, missing source, metric field, and source mismatch rejections unchanged.

- [ ] **Step 4: Run readpage tests**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_readpage_fact_extractor_agent.py -q -o addopts=""
```

Expected: PASS.

---

## Task 3: Add Evidence Curator Agent

**Files:**
- Create: `current_rag_pipeline/rag_pipeline/agents/evidence_curator_agent.py`
- Test: `current_rag_pipeline/tests/test_evidence_curator_agent.py`

- [ ] **Step 1: Write failing tests for deterministic curation**

Create `current_rag_pipeline/tests/test_evidence_curator_agent.py`:

```python
from rag_pipeline.agents.evidence_curator_agent import curate_evidence_batch


def test_curator_keeps_directional_media_signal_with_boundary():
    evidence = [
        {
            "evidence_id": "EV-1",
            "fact": "中证网报道称，Omdia数据显示2025年中国厂商在人形机器人出货量方面领先。",
            "source_level": "C",
            "allowed_use": "directional_signal",
            "source": {"url": "https://example.com/news", "title": "中证网报道"},
        }
    ]

    result = curate_evidence_batch(evidence, query="中国人形机器人产业商业化机会与风险分析")

    assert result["status"] == "ready"
    assert result["curated_evidence_count"] == 1
    note = result["curated_evidence"][0]
    assert note["evidence_id"] == "EV-1"
    assert note["can_support_claim"] is True
    assert note["claim_strength_hint"] == "directional"
    assert "directional" in note["evidence_use_level"]
    assert note["limitations"]


def test_curator_blocks_dirty_login_evidence():
    evidence = [
        {
            "evidence_id": "EV-login",
            "fact": "Login Cookie Privacy Policy Please enable JavaScript",
            "source": {"url": "https://example.com/login"},
        }
    ]

    result = curate_evidence_batch(evidence, query="中国人形机器人")

    assert result["curated_evidence_count"] == 0
    assert result["dirty_blocked_count"] == 1
    assert result["dirty_blocked"][0]["dirty_gate"]["status"] == "blocked"


def test_curator_dedupes_same_fact_by_text_and_source():
    evidence = [
        {
            "evidence_id": "EV-a",
            "fact": "2025年全球人形机器人出货约1.3万台，中国占90%份额。",
            "source": {"url": "https://example.com/a"},
        },
        {
            "evidence_id": "EV-b",
            "fact": "2025年全球人形机器人出货约1.3万台，中国占90%份额。",
            "source": {"url": "https://example.com/a"},
        },
    ]

    result = curate_evidence_batch(evidence, query="中国人形机器人")

    assert result["curated_evidence_count"] == 1
    assert result["deduped_count"] == 1
    assert result["curated_evidence"][0]["merged_evidence_ids"] == ["EV-a", "EV-b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_evidence_curator_agent.py -q -o addopts=""
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement deterministic curator first**

Create `current_rag_pipeline/rag_pipeline/agents/evidence_curator_agent.py`:

```python
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Sequence

from rag_pipeline.contracts.evidence_dirty_gate import evaluate_dirty_gate


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _compact(value: Any, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].rstrip()


def _source(item: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(item.get("source"))


def _source_url(item: Dict[str, Any]) -> str:
    source = _source(item)
    return str(item.get("source_url") or source.get("url") or source.get("source_url") or "").strip()


def _source_level(item: Dict[str, Any]) -> str:
    return str(item.get("source_level") or _source(item).get("source_level") or "C").strip().upper() or "C"


def _fact_text(item: Dict[str, Any]) -> str:
    return _compact(
        item.get("clean_fact")
        or item.get("distilled_fact")
        or item.get("fact")
        or item.get("content")
        or item.get("evidence"),
        520,
    )


def _dedupe_key(item: Dict[str, Any]) -> str:
    text = re.sub(r"\s+", "", _fact_text(item)).lower()
    source = _source_url(item).lower().rstrip("/")
    digest = hashlib.sha1(f"{source}|{text}".encode("utf-8")).hexdigest()
    return digest


def _fact_type(item: Dict[str, Any]) -> str:
    text = " ".join(
        str(value or "")
        for value in (
            item.get("proof_role"),
            item.get("analysis_role"),
            item.get("metric"),
            item.get("fact_type"),
            _fact_text(item),
        )
    )
    if re.search(r"市场|规模|出货|份额|增速|价格|成本|订单|亿元|万台|%", text):
        return "market_signal"
    if re.search(r"政策|监管|标准|目录|补贴|方案", text):
        return "policy_signal"
    if re.search(r"风险|失败|下滑|取消|约束|瓶颈|挑战", text):
        return "risk_signal"
    if re.search(r"客户|订单|采购|中标|部署|应用|案例|场景", text):
        return "case_signal"
    if re.search(r"技术|算法|控制|传感器|续航|供应链|国产化", text):
        return "technology_signal"
    return "context_signal"


def _usable_for(fact_type: str) -> List[str]:
    mapping = {
        "market_signal": ["market_size", "competition", "commercialization"],
        "policy_signal": ["policy", "risk", "commercialization"],
        "risk_signal": ["risk", "boundary", "counter"],
        "case_signal": ["case", "demand", "commercialization"],
        "technology_signal": ["technology", "supply_chain", "commercialization"],
        "context_signal": ["background", "trend"],
    }
    return mapping.get(fact_type, ["background"])


def _claim_strength_hint(item: Dict[str, Any], fact_type: str) -> str:
    level = _source_level(item)
    allowed = str(item.get("allowed_use") or "").strip().lower()
    if allowed == "directional_signal" or level in {"C", "D"}:
        return "directional"
    if level in {"A", "B"} and fact_type in {"market_signal", "policy_signal", "case_signal", "technology_signal"}:
        return "moderate"
    return "directional"


def _limitations(item: Dict[str, Any]) -> List[str]:
    limitations: List[str] = []
    level = _source_level(item)
    if level in {"C", "D"}:
        limitations.append("来源可用于方向性分析，不应单独支撑强结论。")
    metric = str(item.get("metric") or "").strip()
    if metric and (not str(item.get("unit") or "").strip() or not str(item.get("period") or "").strip()):
        limitations.append("指标字段不完整，正文需说明口径或使用谨慎措辞。")
    if str(item.get("source_level_gap") or "").strip():
        limitations.append("来源等级低于原任务偏好，建议作为辅助信号。")
    return limitations or ["需要结合其他证据交叉判断。"]


def _curated_note(item: Dict[str, Any], merged_ids: List[str]) -> Dict[str, Any]:
    fact_type = _fact_type(item)
    strength = _claim_strength_hint(item, fact_type)
    source = _source(item)
    return {
        "schema_version": "curated_evidence_v1",
        "evidence_id": str(item.get("evidence_id") or merged_ids[0] or "").strip(),
        "merged_evidence_ids": merged_ids,
        "source_id": str(item.get("source_id") or item.get("run_source_id") or source.get("source_id") or "").strip(),
        "source_url": _source_url(item),
        "source_title": _compact(source.get("title") or item.get("source_title"), 160),
        "source_level": _source_level(item),
        "clean_fact": _fact_text(item),
        "fact_type": fact_type,
        "usable_for": _usable_for(fact_type),
        "evidence_use_level": "directional_signal" if strength == "directional" else "analysis_signal",
        "claim_strength_hint": strength,
        "can_support_claim": True,
        "limitations": _limitations(item),
        "must_not_use_as": ["official_statistic"] if strength == "directional" else [],
        "dirty": False,
        "lineage": _as_dict(item.get("lineage")),
        "requirement_id": str(item.get("requirement_id") or "").strip(),
        "chapter_id": str(item.get("chapter_id") or _as_dict(item.get("lineage")).get("chapter_id") or "").strip(),
        "proof_role": str(item.get("proof_role") or "").strip(),
    }


def curate_evidence_batch(
    evidence_items: Sequence[Dict[str, Any]],
    *,
    query: str = "",
    max_items: int = 240,
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    dirty_blocked: List[Dict[str, Any]] = []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        dirty_gate = evaluate_dirty_gate(item)
        if dirty_gate["status"] == "blocked":
            dirty_blocked.append({"evidence_id": item.get("evidence_id"), "dirty_gate": dirty_gate})
            continue
        key = _dedupe_key(item)
        groups.setdefault(key, []).append(item)

    curated: List[Dict[str, Any]] = []
    deduped_count = 0
    for grouped_items in groups.values():
        representative = max(
            grouped_items,
            key=lambda item: (
                1 if _source_level(item) in {"A", "B"} else 0,
                len(_fact_text(item)),
            ),
        )
        merged_ids = [
            str(item.get("evidence_id") or "").strip()
            for item in grouped_items
            if str(item.get("evidence_id") or "").strip()
        ]
        deduped_count += max(0, len(merged_ids) - 1)
        note = _curated_note(representative, merged_ids)
        if note["clean_fact"]:
            curated.append(note)

    curated.sort(
        key=lambda item: (
            1 if item["source_level"] in {"A", "B"} else 0,
            1 if item["claim_strength_hint"] == "moderate" else 0,
            len(item["clean_fact"]),
        ),
        reverse=True,
    )
    curated = curated[:max_items]
    return {
        "schema_version": "evidence_curator_result_v1",
        "status": "ready" if curated else "insufficient",
        "query": query,
        "input_count": len([item for item in evidence_items if isinstance(item, dict)]),
        "curated_evidence_count": len(curated),
        "dirty_blocked_count": len(dirty_blocked),
        "deduped_count": deduped_count,
        "curated_evidence": curated,
        "dirty_blocked": dirty_blocked[:20],
    }
```

- [ ] **Step 4: Run curator tests**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_evidence_curator_agent.py -q -o addopts=""
```

Expected: PASS.

---

## Task 4: Wire Curated Evidence Into Analysis Input

**Files:**
- Modify: `current_rag_pipeline/rag_pipeline/agents/analysis_agent.py`
- Test: `current_rag_pipeline/tests/test_analysis_main_chain_contract.py`

- [ ] **Step 1: Add failing test for curated evidence preference**

Append to `current_rag_pipeline/tests/test_analysis_main_chain_contract.py`:

```python
def test_llm_analysis_input_prefers_curated_evidence_notes(monkeypatch):
    from rag_pipeline.agents.analysis_agent import build_llm_analysis_input_v2

    monkeypatch.setenv("REPORT_ANALYSIS_USE_CURATED_EVIDENCE", "true")
    package = {
        "query": "中国人形机器人产业商业化机会与风险分析",
        "curated_evidence": {
            "status": "ready",
            "curated_evidence": [
                {
                    "evidence_id": "CE-1",
                    "chapter_id": "ch_01",
                    "clean_fact": "Omdia数据显示，2025年全球人形机器人出货约1.3万台，中国占90%份额。",
                    "fact_type": "market_signal",
                    "claim_strength_hint": "directional",
                    "evidence_use_level": "directional_signal",
                    "source_id": "SRC-1",
                    "source_level": "C",
                    "source_url": "https://example.com/omdia",
                    "limitations": ["机构估算，非官方统计。"],
                }
            ],
        },
        "chapter_evidence_diagnostics": {
            "ch_01": {
                "chapter_id": "ch_01",
                "chapter_title": "市场空间与商业化节奏",
                "chapter_question": "市场是否真实存在",
            }
        },
    }

    result = build_llm_analysis_input_v2(package, {"query": package["query"]})

    assert result["chapters"]
    cards = result["chapters"][0]["fact_cards"]
    assert cards[0]["evidence_id"] == "CE-1"
    assert cards[0]["distilled_fact"].startswith("Omdia数据显示")
    assert cards[0]["allowed_use"] == "directional_signal"
    assert cards[0]["limitations"] == ["机构估算，非官方统计。"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_analysis_main_chain_contract.py::test_llm_analysis_input_prefers_curated_evidence_notes -q -o addopts=""
```

Expected: FAIL because `build_llm_analysis_input_v2` does not read `curated_evidence`.

- [ ] **Step 3: Add curated-card builder in `analysis_agent.py`**

Add helper functions near `_evidence_cards_for_llm`:

```python
def _curated_evidence_cards_for_llm(
    evidence_package: Dict[str, Any],
    *,
    max_chapters: int,
    max_per_chapter: int,
) -> List[Dict[str, Any]]:
    if not _env_flag("REPORT_ANALYSIS_USE_CURATED_EVIDENCE", True):
        return []
    curated_payload = _as_dict(evidence_package.get("curated_evidence"))
    notes = [
        item
        for item in _as_list(curated_payload.get("curated_evidence"))
        if isinstance(item, dict) and str(item.get("evidence_id") or "").strip()
    ]
    if not notes:
        return []
    chapter_filter = _chapter_filter_for_llm(evidence_package, max_chapters=max_chapters)
    buckets: Dict[str, int] = {}
    cards: List[Dict[str, Any]] = []
    for note in notes:
        raw_chapter_id = str(note.get("chapter_id") or _as_dict(note.get("lineage")).get("chapter_id") or "").strip()
        chapter_id = raw_chapter_id
        if chapter_filter:
            resolved = resolve_chapter_id(chapter_filter, raw_chapter_id)
            if not resolved:
                continue
            chapter_id = resolved
        if not chapter_id:
            chapter_id = "curated_evidence"
        if buckets.get(chapter_id, 0) >= max_per_chapter:
            continue
        cards.append(
            {
                "evidence_id": str(note.get("evidence_id") or "").strip(),
                "chapter_id": chapter_id,
                "requirement_id": str(note.get("requirement_id") or "").strip(),
                "analysis_role": str(note.get("fact_type") or "").strip(),
                "analysis_eligible": True,
                "allowed_use": str(note.get("evidence_use_level") or "directional_signal").strip(),
                "source_id": str(note.get("source_id") or "").strip(),
                "search_task_id": str(note.get("search_task_id") or "").strip(),
                "lineage": _as_dict(note.get("lineage")),
                "distilled_fact": _compact(note.get("clean_fact"), _env_int("BRAIN_LLM_ANALYSIS_MAX_FACT_CHARS", 420, min_value=40, max_value=800)),
                "fact": _compact(note.get("clean_fact"), 420),
                "fact_type": str(note.get("fact_type") or "").strip(),
                "source_level": str(note.get("source_level") or "").strip().upper(),
                "proof_role": str(note.get("proof_role") or note.get("fact_type") or "").strip().lower(),
                "source_verification_status": "curated",
                "claim_strength_hint": str(note.get("claim_strength_hint") or "directional").strip(),
                "limitations": _as_list(note.get("limitations")),
                "usable_for": _as_list(note.get("usable_for")),
                "source_title": _compact(note.get("source_title"), 120),
                "source_url": str(note.get("source_url") or "").strip(),
            }
        )
        buckets[chapter_id] = buckets.get(chapter_id, 0) + 1
    return cards
```

At the start of `_evidence_cards_for_llm`, before ledger fallback, add:

```python
    curated_cards = _curated_evidence_cards_for_llm(
        evidence_package,
        max_chapters=max_chapters,
        max_per_chapter=max_per_chapter,
    )
    if curated_cards:
        return curated_cards
```

- [ ] **Step 4: Run targeted test**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_analysis_main_chain_contract.py::test_llm_analysis_input_prefers_curated_evidence_notes -q -o addopts=""
```

Expected: PASS.

---

## Task 5: Wire Curator Stage Into Full Report Flow

**Files:**
- Modify: `current_rag_pipeline/rag_pipeline/flows/report/full_report.py`
- Test: `current_rag_pipeline/tests/test_artifact_pipeline_bridge.py` or a new focused test if `full_report` has existing flow-level helpers.

- [ ] **Step 1: Add flow-level helper test**

Create or append a focused test:

```python
def test_attach_curated_evidence_to_evidence_package(monkeypatch):
    from rag_pipeline.flows.report import full_report

    monkeypatch.setenv("REPORT_EVIDENCE_CURATOR_ENABLED", "true")
    evidence_package = {
        "query": "中国人形机器人",
        "clean_evidence_list": [
            {
                "evidence_id": "EV-1",
                "fact": "Omdia数据显示，2025年全球人形机器人出货约1.3万台，中国占90%份额。",
                "source": {"url": "https://example.com/omdia"},
                "source_level": "C",
            }
        ],
    }

    result = full_report._attach_curated_evidence(evidence_package)

    assert result is not evidence_package
    assert result["curated_evidence"]["status"] == "ready"
    assert result["curated_evidence"]["curated_evidence_count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_artifact_pipeline_bridge.py::test_attach_curated_evidence_to_evidence_package -q -o addopts=""
```

Expected: FAIL because `_attach_curated_evidence` does not exist or test file target needs adjustment.

- [ ] **Step 3: Implement helper in `full_report.py`**

Add near other pipeline helpers:

```python
def _attach_curated_evidence(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    if not _env_flag("REPORT_EVIDENCE_CURATOR_ENABLED", True):
        return dict(evidence_package or {})
    try:
        from rag_pipeline.agents.evidence_curator_agent import curate_evidence_batch
    except Exception:
        return dict(evidence_package or {})

    package = dict(evidence_package or {})
    candidates = (
        _as_list(package.get("analysis_ready_evidence"))
        + _as_list(package.get("clean_evidence_list"))
        + _as_list(package.get("supporting_evidence"))
    )
    curated = curate_evidence_batch(
        [item for item in candidates if isinstance(item, dict)],
        query=str(package.get("query") or ""),
        max_items=_env_int("REPORT_EVIDENCE_CURATOR_MAX_ITEMS", 240, min_value=20, max_value=1000),
    )
    package["curated_evidence"] = curated
    return package
```

Call `_attach_curated_evidence(evidence_package)` after evidence merge and before `run_analysis_agent` / `analysis_agent` receives the package.

- [ ] **Step 4: Run flow test**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_artifact_pipeline_bridge.py::test_attach_curated_evidence_to_evidence_package -q -o addopts=""
```

Expected: PASS after adjusting the test location if needed.

---

## Task 6: Make Analysis Prompt Use Curated Evidence Inventory

**Files:**
- Modify: `current_rag_pipeline/rag_pipeline/agents/analysis_agent.py`
- Test: `current_rag_pipeline/tests/test_analysis_main_chain_contract.py`

- [ ] **Step 1: Add prompt contract test**

Append:

```python
def test_llm_chapter_prompt_emphasizes_inventory_before_claims():
    from rag_pipeline.agents import analysis_agent

    prompt = analysis_agent._llm_chapter_system_prompt()

    assert "evidence inventory" in prompt.lower()
    assert "convert relevant weak evidence into bounded directional claims" in prompt.lower()
    assert "do not reject evidence only because source level is c or d" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_analysis_main_chain_contract.py::test_llm_chapter_prompt_emphasizes_inventory_before_claims -q -o addopts=""
```

Expected: FAIL because prompt does not contain those exact contract phrases.

- [ ] **Step 3: Update `_llm_chapter_system_prompt`**

Add this block before the claim unit schema:

```text
First build an evidence inventory mentally from the supplied fact_cards:
- identify market, policy, player, order, price, case, technology, supply-chain and risk signals;
- merge duplicated evidence that points to the same fact;
- keep source limitations as boundaries instead of deleting the evidence;
- do not reject evidence only because source level is C or D;
- convert relevant weak evidence into bounded directional claims when it has a traceable source.
```

Also add:

```text
When evidence is traceable but weak, produce a directional or limited_evidence claim with explicit limitation_boundary.
Do not over-expand a fact into a stronger mechanism than the cited evidence supports.
Prefer more bounded claims over fewer over-strong claims.
```

- [ ] **Step 4: Bump LLM analysis prompt version**

In `synthesize_chapter_with_llm_analysis`, change:

```python
"prompt_version": "llm_analysis_v3_2026_06_typed_claims",
```

to:

```python
"prompt_version": "llm_analysis_v4_2026_06_curated_inventory",
```

- [ ] **Step 5: Run prompt contract test**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_analysis_main_chain_contract.py::test_llm_chapter_prompt_emphasizes_inventory_before_claims -q -o addopts=""
```

Expected: PASS.

---

## Task 7: Add Curated Conversion Monitoring

**Files:**
- Modify: `current_rag_pipeline/rag_pipeline/observability/evidence_claim_conversion.py`
- Test: `current_rag_pipeline/tests/test_evidence_claim_conversion.py`

- [ ] **Step 1: Add failing metric test**

Append:

```python
def test_conversion_monitor_reports_curated_to_claim_rate():
    from rag_pipeline.observability.evidence_claim_conversion import summarize_evidence_claim_conversion

    writer_package = {
        "evidence_package": {
            "curated_evidence": {
                "curated_evidence": [
                    {"evidence_id": "CE-1"},
                    {"evidence_id": "CE-2"},
                ]
            }
        },
        "structured_analysis": {
            "claim_units": [
                {"claim_id": "CL-1", "fact_ids": ["CE-1"]},
            ]
        },
    }

    result = summarize_evidence_claim_conversion(writer_package)

    assert result["curated_evidence_count"] == 2
    assert result["curated_evidence_used_in_claim_count"] == 1
    assert result["curated_to_claim_rate"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_evidence_claim_conversion.py::test_conversion_monitor_reports_curated_to_claim_rate -q -o addopts=""
```

Expected: FAIL because metrics are not present.

- [ ] **Step 3: Implement curated metrics**

Inside `summarize_evidence_claim_conversion`, compute:

```python
curated_items = _as_list(_as_dict(_as_dict(writer_package.get("evidence_package")).get("curated_evidence")).get("curated_evidence"))
curated_ids = {
    str(item.get("evidence_id") or "").strip()
    for item in curated_items
    if isinstance(item, dict) and str(item.get("evidence_id") or "").strip()
}
claim_fact_ids = {
    str(ref or "").strip()
    for claim in _as_list(_as_dict(writer_package.get("structured_analysis")).get("claim_units"))
    if isinstance(claim, dict)
    for ref in _as_list(claim.get("fact_ids") or claim.get("used_evidence_ids") or claim.get("evidence_refs"))
    if str(ref or "").strip()
}
curated_used = curated_ids & claim_fact_ids
summary["curated_evidence_count"] = len(curated_ids)
summary["curated_evidence_used_in_claim_count"] = len(curated_used)
summary["curated_to_claim_rate"] = round(len(curated_used) / max(len(curated_ids), 1), 4) if curated_ids else 0.0
```

- [ ] **Step 4: Run conversion tests**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_evidence_claim_conversion.py -q -o addopts=""
```

Expected: PASS.

---

## Task 8: Full Verification

**Files:**
- No new files.

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
python -m pytest current_rag_pipeline/tests/test_evidence_dirty_gate.py current_rag_pipeline/tests/test_evidence_curator_agent.py current_rag_pipeline/tests/test_readpage_fact_extractor_agent.py current_rag_pipeline/tests/test_analysis_main_chain_contract.py current_rag_pipeline/tests/test_evidence_claim_conversion.py -q -o addopts=""
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run:

```powershell
python -m pytest current_rag_pipeline/tests -q -o addopts=""
```

Expected: all tests pass.

- [ ] **Step 3: Run one monitored live task**

Use a topic with known public evidence:

```powershell
python -m current_rag_pipeline.run_full_report `
  --query "2026年中国人形机器人产业商业化机会与风险分析" `
  --quality-mode high `
  --llm-profile qwen
```

Expected monitoring shifts:

- `curated_evidence_count` is greater than `claim_unit_count`.
- `curated_to_claim_rate` is measurable.
- `source_level_below_required` no longer dominates readpage rejection reasons.
- `analysis_ready_count` and `curated_evidence_count` do not collapse.
- `claim_unit_count` increases or stays stable while semantic partial issues decrease.
- Report body length improves without table dependence.

---

## Self-Review

- Spec coverage: The plan covers dirty-data-only hard blocking, model/deterministic curation, relaxed readpage source-level handling, curated analysis input, prompt update, and conversion monitoring.
- Placeholder scan: No TBD/TODO placeholders are present. Each task has concrete files, test code, commands, and expected outcomes.
- Type consistency: The central payload names are consistent: `curated_evidence`, `curated_evidence_count`, `curated_to_claim_rate`, `evidence_use_level`, `claim_strength_hint`, and `limitations`.
- Scope check: This plan intentionally does not restore tables, does not tune scoring, and does not optimize artifact ledger/cache write amplification. It focuses only on evidence-to-analysis recall.
