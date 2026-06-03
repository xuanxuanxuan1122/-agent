# Public Report Quality Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining gap between “audit blockers removed” and “public report reads like a polished research report.”

**Architecture:** Keep the current LLM analysis, citation manifest, composer, and sanitizer pipeline. Move diagnostic tables out of the public appendix, make narrative leakage auditing stricter instead of hiding framing sentences from citation audit only, and update body rewrite/P4 orchestration tests to match the new independent-control behavior.

**Tech Stack:** Python, pytest, existing `rag_pipeline.agents` modules, existing offline fixture `output/full_reports/20260520_161059_AI_Agent生态发展报告：从工具到智能体的范式跃迁_clean.md`.

---

## Current Findings

- `tools/test_sanitizer_fixes.py` passes only with `PYTHONIOENCODING=utf-8`; on default Windows GBK it crashes while printing `•`.
- Sanitizer/audit now removes hard blockers from the historical clean report: diagnostic line count drops to 0, `Competitive signal` drops to 0, and citationless factual segments drop to 0.
- The cleaned preview still contains soft process language such as “事实锚点”, “事实起点”, “后续重点跟踪”, “这些事实来自不同类型来源且方向一致时”. These are excluded by `NON_FACTUAL_FRAMING_RE`, so they no longer block citation audit, but they still read like internal analysis scaffolding.
- `markdown_renderer.render_appendix()` starts with `## 来源附录`, renders metric/diagnostic appendix tables under that heading, then emits a second `## 来源附录` for actual sources. This lets diagnostic tables bypass the public body sanitizer because they are classified as appendix content.
- `table_agent.py` still produces diagnostic table labels and phrases such as `Competitive signal`, `Risk boundary`, `后续影响`, `使用边界`, and placeholder implications. Current public cleanup relies on downstream rewriting/removal.
- `tests/test_report_contracts_and_composer.py::test_chapter_argument_skips_section_body_rewrite_when_chapter_narrative_enabled` is stale. The implementation intentionally removed that gate so section rewrite can still run when P4 is enabled but skipped.

---

## Task 1: Make the offline sanitizer driver Windows-safe

**Files:**
- Modify: `D:/pychram/RAG2/current_rag_pipeline/tools/test_sanitizer_fixes.py`
- Test: manual command

- [ ] **Step 1: Replace Unicode bullets in console output**

Change every direct `"•"` console marker to an ASCII marker such as `"-"`, or set `errors="replace"` on stdout writes.

Expected pattern:

```python
print(f"  - {sample}")
```

- [ ] **Step 2: Run the driver without PYTHONIOENCODING**

Run:

```powershell
python tools\test_sanitizer_fixes.py
```

Expected: exit code 0, no `UnicodeEncodeError`.

- [ ] **Step 3: Run the driver with UTF-8 too**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; python tools\test_sanitizer_fixes.py
```

Expected: exit code 0 and `after: count=0` for citationless factual segments.

---

## Task 2: Move diagnostic appendix tables out of the public report

**Files:**
- Modify: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/markdown_renderer.py`
- Modify if needed: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/final_writer_agent.py`
- Test: `D:/pychram/RAG2/current_rag_pipeline/tests/test_markdown_renderer_naturalness.py`

- [ ] **Step 1: Add a regression test for source appendix purity**

Add a test that calls `render_appendix()` with `metric_normalization_table` and `table_appendix_rows` containing diagnostic language.

Expected assertions:

```python
assert rendered.count("## 来源附录") == 1
assert "附录明细" not in rendered
assert "指标口径表" not in rendered
assert "后续影响" not in rendered
assert "- [1]" in rendered
```

- [ ] **Step 2: Change `render_appendix()` to render only sources by default**

Public `_report.md` should only receive source rows. Move metric normalization tables, coverage matrix, and table appendix details to score diagnostics or behind explicit opt-in flags.

Recommended default:

```python
if not source_registry:
    return ""
lines = ["## 来源附录"]
for source in list(source_registry)[:50]:
    ...
return "\n".join(lines)
```

Keep diagnostic appendix rendering only behind explicit environment flags such as `REPORT_RENDER_COVERAGE_MATRIX` and a new `REPORT_RENDER_DIAGNOSTIC_APPENDIX_TABLES=false`.

- [ ] **Step 3: Verify FinalAudit source heading compatibility**

Run:

```powershell
python -m pytest tests\test_final_audit_agent.py tests\test_markdown_renderer_naturalness.py -q
```

Expected: source appendix is detected and diagnostic appendix tables do not appear in public markdown.

---

## Task 3: Treat internal framing language as public narrative leakage, not only non-factual audit noise

**Files:**
- Modify: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/public_report_sanitizer.py`
- Modify: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/report_contracts.py`
- Test: `D:/pychram/RAG2/current_rag_pipeline/tests/test_markdown_renderer_naturalness.py`

- [ ] **Step 1: Add public leak tests for framing phrases**

Create assertions that the final sanitized report does not include:

```python
forbidden = [
    "事实锚点",
    "事实起点",
    "后续重点跟踪",
    "可复核材料指向",
    "这些事实来自不同类型来源且方向一致时",
    "来源集中、口径不一致",
    "待验证方向",
    "尚不足以支撑强结论",
]
```

- [ ] **Step 2: Extend `PUBLIC_NARRATIVE_BLOCK_PATTERNS`**

Add these framing phrases as `writing_process_language` or `analysis_scaffold_language` blockers.

Important: do not only add them to `NON_FACTUAL_FRAMING_RE`; they should be removed or rewritten from `_report.md`, not merely ignored by citation audit.

- [ ] **Step 3: Add deterministic rewrites only when safe**

Safe rewrites:

```text
“可复核材料指向：X” -> “公开材料显示，X”
“这些事实来自不同类型来源且方向一致时...” -> drop sentence
“后续重点跟踪...” -> drop sentence
“事实锚点/事实起点” -> drop sentence or rewrite to a direct judgment only if citation is present
```

- [ ] **Step 4: Re-run offline sanitizer driver**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; python tools\test_sanitizer_fixes.py
```

Expected: no hard blockers, no framing phrases in the after preview, citationless factual count remains 0.

---

## Task 4: Update body rewrite / P4 orchestration tests to match current design

**Files:**
- Modify: `D:/pychram/RAG2/current_rag_pipeline/tests/test_report_contracts_and_composer.py`
- Test: same file

- [ ] **Step 1: Replace stale skip test**

Rename `test_chapter_argument_skips_section_body_rewrite_when_chapter_narrative_enabled`.

New behavior: P4 enabled must not automatically disable section body rewrite. Section rewrite is controlled by `REPORT_ENABLE_LLM_BODY_REWRITE`.

Expected assertion:

```python
monkeypatch.setenv("REPORT_ENABLE_LLM_CHAPTER_NARRATIVE", "true")
monkeypatch.setenv("REPORT_ENABLE_LLM_BODY_REWRITE", "true")
...
assert section["body_rewrite_status"] == "rewritten"
```

- [ ] **Step 2: Add explicit off test**

When `REPORT_ENABLE_LLM_BODY_REWRITE=false`, rewrite should not run even if P4 is enabled.

Expected:

```python
assert "body_rewrite_status" not in section
```

- [ ] **Step 3: Run targeted tests**

Run:

```powershell
python -m pytest tests\test_report_contracts_and_composer.py tests\test_section_body_rewrite_agent.py tests\test_chapter_narrative_agent.py -q
```

Expected: all pass.

---

## Task 5: Reduce upstream diagnostic table language at source

**Files:**
- Modify: `D:/pychram/RAG2/current_rag_pipeline/rag_pipeline/agents/table_agent.py`
- Test: add or update table tests

- [ ] **Step 1: Add a public table row test**

Given a public table row, assert it does not contain:

```python
["Competitive signal", "Risk boundary", "后续影响", "使用边界", "该信号只有与反例和高等级来源同向时"]
```

- [ ] **Step 2: Replace public-facing headers/fields**

For public tables:

```python
"Competitive signal" -> "竞争信号"
"Risk boundary" -> "风险边界"
"后续影响" -> "判断含义" or omit
"使用边界" -> "适用范围" or omit
```

- [ ] **Step 3: Keep diagnostic wording in score-only structures**

If the table is intended for diagnostics, set `appendix_only=True` or a score-only flag so it never reaches `_report.md`.

---

## Task 6: Full verification

**Files:**
- No direct code changes

- [ ] **Step 1: Compile changed modules**

Run:

```powershell
python -m py_compile rag_pipeline\agents\public_report_sanitizer.py rag_pipeline\agents\report_contracts.py rag_pipeline\agents\markdown_renderer.py rag_pipeline\agents\final_writer_agent.py rag_pipeline\agents\chapter_argument_agent.py rag_pipeline\agents\table_agent.py
```

Expected: exit code 0.

- [ ] **Step 2: Run targeted regression tests**

Run:

```powershell
python -m pytest tests\test_citation_manifest.py tests\test_markdown_renderer_naturalness.py tests\test_chapter_narrative_agent.py tests\test_report_contracts_and_composer.py tests\test_section_body_rewrite_agent.py -q --tb=short
```

Expected: all pass.

- [ ] **Step 3: Run default current-contract suite**

Run:

```powershell
python -m pytest tests -q -m "not slow_integration and not legacy_clean_contract" --tb=short
```

Expected: all pass.

- [ ] **Step 4: Replay quality sample**

Run quality replay with the latest AI Agent snapshot and verify:

```text
public_narrative_leak_remaining_count = 0
factual_body_without_citations_count = 0
final_citation_status_after_render = ok
no diagnostic appendix tables in _report.md
body_rewrite runs when enabled even if P4 is enabled/skipped
```
