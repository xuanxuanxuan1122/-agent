# Public Report Green Quality v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the report pipeline from “hard audit blockers removed” to “public report reads like a clean, reader-facing industry research draft.”

**Architecture:** Keep the current high-quality main chain: bounded search/readpage, fact cards, per-chapter LLM analysis, claim-layout matching, deterministic composer, optional body rewrite/P4, citation manifest, FinalAudit. This plan tightens the public rendering boundary: diagnostic artifacts go to `_score.md`, public body keeps only reader-facing sections and sources, and P4/body rewrite are made observable and mutually safe.

**Tech Stack:** Python, pytest, existing `rag_pipeline.agents` modules, local replay/trace tools, no new external dependency.

---

## Current State

The latest fixes prove the following:

- `sanitize_public_markdown()` can reduce the historical `_clean.md` hard blockers to zero.
- `finalize_markdown_citations()` can remove unresolved citations and detect citationless factual lines.
- `chapter_argument_agent` no longer hard-disables section body rewrite just because P4 is enabled.
- However, public reports can still look like processing drafts because renderer/table sources still emit diagnostic blocks and framing language.

Remaining quality risks:

- `render_appendix()` mixes public source appendix and diagnostic appendix tables under `## 来源附录`.
- `table_agent.py` still creates public-looking rows with diagnostic language such as `后续影响`, `使用边界`, `Competitive signal`, `Risk boundary`.
- `NON_FACTUAL_FRAMING_RE` prevents some process-language sentences from blocking citation audit, but those sentences can remain visible in the report.
- Existing tests still include one stale contract expecting section body rewrite to be skipped when P4 is enabled.
- QA/judge inputs can still grow too large if full packages leak into the judge payload.

---

## Task 1: Split Public Appendix from Diagnostic Appendix

**Files:**
- Modify: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/markdown_renderer.py`
- Modify: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/final_writer_agent.py`
- Test: `D:/pychram/RAG2/current_rag_pipeline/tests/test_markdown_renderer_naturalness.py`
- Test: `D:/pychram/RAG2/current_rag_pipeline/tests/test_citation_manifest.py`

- [ ] **Step 1: Add a failing public appendix test**

Add a test that builds an appendix package with `metric_normalization_table` and `table_appendix_rows` containing diagnostic wording.

Expected behavior:

```python
rendered = render_appendix(
    [{"ref": "[1]", "title": "来源A", "url": "https://example.org/a"}],
    {
        "metric_normalization_table": [
            {"metric_name": "CAGR", "subject": "AI Agent", "period": "2028年", "value": "41%", "unit": "%"}
        ],
        "table_appendix_rows": [
            {
                "title": "市场指标与口径表",
                "headers": ["指标", "范围", "期间", "数值", "单位", "后续影响"],
                "rows": [
                    ["CAGR", "全球", "2028年", "41", "%", "该指标须同时披露范围、期间、单位与来源等级,才进入正文判断。"],
                    ["市场规模", "全球", "2028年", "100", "亿美元", "该指标须同时披露范围、期间、单位与来源等级,才进入正文判断。"],
                ],
                "should_render": True,
                "validation_status": "passed",
            }
        ],
    },
)
assert rendered.count("## 来源附录") == 1
assert "- [1]" in rendered
assert "指标口径表" not in rendered
assert "附录明细" not in rendered
assert "后续影响" not in rendered
assert "该指标须" not in rendered
```

- [ ] **Step 2: Make `render_appendix()` source-only by default**

Change `render_appendix()` so the public path only renders actual sources:

```python
def render_appendix(source_registry, appendix_package):
    if not source_registry:
        return ""
    lines = ["## 来源附录"]
    for source in list(source_registry)[:50]:
        ...
    return "\n".join(lines)
```

Diagnostic tables should not be public by default. If they are still needed, move them to score rendering or guard with an explicit opt-in:

```python
REPORT_RENDER_DIAGNOSTIC_APPENDIX_TABLES=false
```

- [ ] **Step 3: Keep FinalAudit aligned**

Verify FinalAudit detects the rendered source heading:

```powershell
python -m pytest tests\test_final_audit_agent.py tests\test_markdown_renderer_naturalness.py -q
```

Expected: no `missing_sources_appendix` when report contains numbered citations and `## 来源附录`.

---

## Task 2: Promote Framing Language to Public Narrative Blockers

**Files:**
- Modify: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/public_report_sanitizer.py`
- Modify: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/report_contracts.py`
- Test: `D:/pychram/RAG2/current_rag_pipeline/tests/test_markdown_renderer_naturalness.py`
- Test tool: `D:/pychram/RAG2/current_rag_pipeline/tools/test_sanitizer_fixes.py`

- [ ] **Step 1: Add a regression test for process-language leakage**

Add a fixture body containing:

```text
本章判断应以“市场规模: 100亿美元”为事实锚点。
先用 增速: 44.2% 确认本章的事实起点。
后续重点跟踪同口径指标、反向样本和执行进展。
可复核材料指向：某行业报告显示市场规模增长。
这些事实来自不同类型来源且方向一致时，可以支撑较强结论。
来源集中、口径不一致或缺少反向样本时，结论会保留边界。
```

Expected after `sanitize_public_markdown()`:

```python
for phrase in ["事实锚点", "事实起点", "后续重点跟踪", "可复核材料指向", "这些事实来自不同类型来源", "来源集中、口径不一致"]:
    assert phrase not in cleaned
```

- [ ] **Step 2: Extend `PUBLIC_NARRATIVE_BLOCK_PATTERNS`**

Add these patterns to public narrative audit:

```python
(r"事实锚点|事实起点", "analysis_scaffold_language"),
(r"后续重点跟踪|可复核材料指向", "analysis_scaffold_language"),
(r"这些事实来自不同类型来源|来源集中、口径不一致", "analysis_scaffold_language"),
(r"待验证方向|尚不足以支撑强结论", "fallback_claim_language"),
```

- [ ] **Step 3: Rewrite or drop safely**

Rules:

- Drop sentences with `事实锚点`, `事实起点`, `后续重点跟踪`.
- Rewrite `可复核材料指向：X` to `公开材料显示，X` only if the line already has a citation.
- Drop `这些事实来自不同类型来源...` and `来源集中、口径不一致...`; these are meta-analysis, not public prose.

- [ ] **Step 4: Keep citation audit strict**

Do not rely on `NON_FACTUAL_FRAMING_RE` to hide public leaks. It can remain to avoid false citation blockers, but public narrative gate must remove those lines.

- [ ] **Step 5: Re-run offline driver**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; python tools\test_sanitizer_fixes.py
```

Expected:

- `after: count=0` for real citationless factual segments.
- no “事实锚点 / 事实起点 / 后续重点跟踪 / 可复核材料指向” in the printed after preview.

---

## Task 3: Update P4 and Section Rewrite Contract Tests

**Files:**
- Modify: `D:/pychram/RAG2/current_rag_pipeline/tests/test_report_contracts_and_composer.py`
- Test: `D:/pychram/RAG2/current_rag_pipeline/tests/test_section_body_rewrite_agent.py`
- Test: `D:/pychram/RAG2/current_rag_pipeline/tests/test_chapter_narrative_agent.py`

- [ ] **Step 1: Replace stale skip test**

The old assertion says section rewrite must be skipped when P4 is enabled. Replace it with:

```python
def test_chapter_argument_body_rewrite_independent_from_chapter_narrative(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
    ...
    assert section["body_rewrite_status"] == "rewritten"
```

- [ ] **Step 2: Add an explicit body rewrite off test**

```python
def test_chapter_argument_body_rewrite_off_even_when_chapter_narrative_enabled(monkeypatch):
    monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")
    monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "false")
    ...
    assert "body_rewrite_status" not in section
```

- [ ] **Step 3: Run targeted tests**

Run:

```powershell
python -m pytest tests\test_report_contracts_and_composer.py tests\test_section_body_rewrite_agent.py tests\test_chapter_narrative_agent.py -q --tb=short
```

Expected: all pass.

---

## Task 4: Reduce Diagnostic Table Language at Source

**Files:**
- Modify: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/table_agent.py`
- Test: create or update table-related tests, preferably `D:/pychram/RAG2/current_rag_pipeline/tests/test_markdown_renderer_naturalness.py`

- [ ] **Step 1: Add source-level table language test**

Construct a table row from the table agent and assert public-facing cells do not contain:

```python
["Competitive signal", "Risk boundary", "后续影响", "使用边界", "该信号只有与反例和高等级来源同向时", "需用高等级来源复核"]
```

- [ ] **Step 2: Replace English/internal labels**

Change public labels:

```python
"Competitive signal" -> "竞争信号"
"Risk boundary" -> "风险边界"
"Maturity signal" -> "成熟信号"
```

- [ ] **Step 3: Separate public rows from score rows**

Rows with `appendix_only=True`, placeholder implications, or incomplete sources should be routed to score diagnostics, not public report tables.

Add or enforce fields:

```python
"public_render": False
"score_only_reason": "diagnostic_table_language"
```

- [ ] **Step 4: Keep public table rows fact-like**

Public table rows should only carry:

- object/subject
- metric/signal
- period/scope
- value/unit if present
- citation/source ref

They should not carry “后续影响 / 使用边界 / 进入判断 / 观察指标 / 验证方法” language.

---

## Task 5: Make QA/Judge Payload Compact by Contract

**Files:**
- Modify: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/rewrite_agent.py`
- Modify if needed: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/qa_agent.py`
- Test: add/update tests for QA compacting

- [ ] **Step 1: Add payload size test**

Build a fake report package with large `chapter_packages`, `evidence_health_summary`, and raw evidence blobs.

Expected:

```python
payload = _compact_qa_for_llm(package)
assert "chapter_packages" not in payload
assert "evidence_health_summary" not in payload
assert len(json.dumps(payload, ensure_ascii=False)) < 30000
```

- [ ] **Step 2: Enforce compact input for LLM judge/rewrite**

Ensure any QA/judge call receives the compact payload, not full writer package.

- [ ] **Step 3: Trace the input size**

Write `qa_llm_input_chars` or token estimate into score/trace so future runs show whether this regresses.

---

## Task 6: Add a Public Report Gate Smoke Test

**Files:**
- Create: `D:/pychram/RAG2/current_rag_pipeline/tests/test_public_report_gate_smoke.py`

- [ ] **Step 1: Test combined rendering + sanitizer**

The fixture should include:

- a body section with valid citation
- one diagnostic appendix table
- one process-language paragraph
- one source appendix entry

Expected:

```python
assert "## 来源附录" in final
assert "附录明细" not in final
assert "指标口径表" not in final
assert "事实锚点" not in final
assert "后续重点跟踪" not in final
assert "该指标须" not in final
assert public_narrative_leak_audit(final)["blocker_count"] == 0
assert len(_citationless_factual_segments(final)) == 0
```

- [ ] **Step 2: Ensure score-only diagnostics are preserved elsewhere**

The test should not require diagnostic data to disappear globally. It only asserts `_report.md` public markdown is clean.

---

## Task 7: Full Verification

**Files:**
- No direct changes

- [ ] **Step 1: Compile changed modules**

Run:

```powershell
python -m py_compile rag_pipeline\agents\public_report_sanitizer.py rag_pipeline\agents\report_contracts.py rag_pipeline\agents\markdown_renderer.py rag_pipeline\agents\final_writer_agent.py rag_pipeline\agents\chapter_argument_agent.py rag_pipeline\agents\table_agent.py rag_pipeline\agents\rewrite_agent.py rag_pipeline\agents\qa_agent.py
```

- [ ] **Step 2: Run targeted tests**

Run:

```powershell
python -m pytest tests\test_citation_manifest.py tests\test_markdown_renderer_naturalness.py tests\test_report_contracts_and_composer.py tests\test_section_body_rewrite_agent.py tests\test_chapter_narrative_agent.py tests\test_public_report_gate_smoke.py -q --tb=short
```

- [ ] **Step 3: Run default suite**

Run:

```powershell
python -m pytest tests -q -m "not slow_integration and not legacy_clean_contract" --tb=short
```

- [ ] **Step 4: Replay quality sample**

Run latest AI Agent quality replay and check:

```text
public_narrative_leak_remaining_count = 0
factual_body_without_citations_count = 0
final_citation_status_after_render = ok
no diagnostic appendix tables in _report.md
no process-language framing in _report.md
body_rewrite runs when enabled even if P4 is enabled/skipped
```

---

## Acceptance Criteria

- Public `_report.md` contains only reader-facing sections and a source appendix.
- Diagnostic tables, metric-mouthful appendix details, and processing language move to `_score.md` or trace.
- `FinalAudit` no longer sees “missing source appendix” caused by malformed appendix structure.
- P4 and section-level body rewrite are independently controlled.
- Historical sanitizer fixture passes on Windows default console and UTF-8 console.
- Default non-slow/non-legacy tests pass.
