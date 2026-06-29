# Report Publishability Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the report pipeline from a 57/100 limited-review draft to a complete publishable draft by fixing remaining citation/source gate, layout binding, and advisory scoring issues without reintroducing hard evidence-quality deletion.

**Architecture:** Keep the current evidence-first, claim-first writer path. Review/audit modules remain diagnostic except for delivery blockers: fake sources, broken citation numbers, internal markers, uncited hard numbers. Publishability scoring should measure the final claim-driven chapters, not unused plan chapters or table expectations.

**Tech Stack:** Python, pytest, existing `rag_pipeline` agents and contracts.

---

### Task 1: Align Publishability Gate With Advisory Evidence Strategy

**Files:**
- Modify: `rag_pipeline/contracts/quality_gate_policy.py`
- Modify: `rag_pipeline/agents/qa_agent.py`
- Modify: `rag_pipeline/flows/report/final_audit_agent.py`
- Test: `tests/test_report_quality_regressions.py`
- Test: `tests/test_final_audit_agent.py`

- [ ] **Step 1: Write failing tests**

Add tests asserting that `publishable_evidence_gate_failed`, A/B source shortage, single-source support, weak-source support, and missing proof standards are warnings/review suggestions in advisory/public-signal mode, not clean blockers.

Run:
```powershell
python -m pytest tests\test_report_quality_regressions.py::test_advisory_evidence_gate_does_not_block_publishable_draft tests\test_final_audit_agent.py::test_final_audit_keeps_weak_source_as_warning -q -o addopts=""
```
Expected: fail before implementation.

- [ ] **Step 2: Implement policy change**

In `quality_gate_policy.py`, classify only these as delivery blockers:
```python
DELIVERY_BLOCKERS = {
    "empty_report",
    "internal_marker_leak",
    "fake_or_placeholder_source",
    "broken_citation_numbering",
    "uncited_hard_number",
    "invalid_url",
}
```
Map these to warnings:
```python
ADVISORY_EVIDENCE_WARNINGS = {
    "publishable_evidence_gate_failed",
    "chapter_core_ab_below_minimum",
    "evidence_preflight_not_ready",
    "public_chapter_without_ab_sources",
    "missing_proof_standards",
    "single_source_support",
    "weak_source_support",
}
```

- [ ] **Step 3: Verify**

Run:
```powershell
python -m pytest tests\test_report_quality_regressions.py tests\test_final_audit_agent.py -q -o addopts=""
```
Expected: pass.

### Task 2: Score Final Claim-Driven Chapters, Not Dropped Plan Chapters

**Files:**
- Modify: `rag_pipeline/flows/report/full_report.py`
- Modify: `rag_pipeline/agents/writer_agent_clean.py`
- Modify: `rag_pipeline/observability/health_metrics.py`
- Test: `tests/test_chapter_recomposer_agent.py`
- Test: `tests/test_report_pipeline_replay.py`

- [ ] **Step 1: Write failing tests**

Create a replay-like package where `plan_chapters` has 8 chapters but `final_chapters` has 6 claim-backed chapters. Assert that health/score does not count the two dropped plan chapters as empty chapters or binding failures.

Run:
```powershell
python -m pytest tests\test_report_pipeline_replay.py::test_score_uses_final_chapters_not_dropped_plan_chapters -q -o addopts=""
```
Expected: fail before implementation.

- [ ] **Step 2: Implement final-chapter health basis**

Where score/health computes `chapter_evidence_binding_failed`, `public_chapter_without_ab_sources`, `layout_validation_failed`, and empty chapter counts, use:
```python
chapters_for_public_scoring = final_chapters or claim_clusters or rendered_chapters
```
Do not count `dropped_plan_chapter_ids` as public body failures.

- [ ] **Step 3: Verify**

Run:
```powershell
python -m pytest tests\test_chapter_recomposer_agent.py tests\test_report_pipeline_replay.py -q -o addopts=""
```
Expected: pass.

### Task 3: Add Citation Repair Before FinalAudit

**Files:**
- Modify: `rag_pipeline/agents/citation_manifest.py`
- Modify: `rag_pipeline/agents/final_writer_agent.py`
- Modify: `rag_pipeline/flows/report/full_report.py`
- Test: `tests/test_citation_manifest.py`
- Test: `tests/test_markdown_renderer_naturalness.py`

- [ ] **Step 1: Write failing tests**

Build a section whose `claim_id` allows refs `[4]` and `[8]`, but rendered text contains `[3]`. Assert that the pre-audit repair removes or replaces `[3]` and reconciles the appendix.

Run:
```powershell
python -m pytest tests\test_citation_manifest.py::test_pre_audit_citation_repair_uses_claim_allowed_refs -q -o addopts=""
```
Expected: fail before implementation.

- [ ] **Step 2: Implement repair**

Add a function such as:
```python
def repair_section_citations_by_claim_refs(markdown, section_claim_ref_map, citation_manifest):
    ...
```
Rules:
- If a paragraph citation is not in the section/claim allowed refs, remove it.
- If the paragraph has no citation after removal and the section has allowed refs, append the best allowed ref.
- Re-run appendix reconciliation after repair.
- Emit diagnostics; do not silently rewrite without trace.

- [ ] **Step 3: Verify**

Run:
```powershell
python -m pytest tests\test_citation_manifest.py tests\test_markdown_renderer_naturalness.py -q -o addopts=""
```
Expected: pass.

### Task 4: Consume Semantic Judge Findings As Writer Caveats

**Files:**
- Modify: `rag_pipeline/agents/claim_builder_agent.py`
- Modify: `rag_pipeline/agents/section_composer.py`
- Test: `tests/test_claim_depth_pack.py`
- Test: `tests/test_report_contracts_and_composer.py`

- [ ] **Step 1: Write failing tests**

Create claim units with semantic judge `partial`, `adjacent`, and `unsupported_observed`. Assert that:
- `partial` keeps the claim but removes overreach terms if provided.
- `adjacent` becomes a contextual observation.
- `unsupported` does not become a core section unless repaired.

Run:
```powershell
python -m pytest tests\test_report_contracts_and_composer.py::test_semantic_judge_findings_become_writer_caveats -q -o addopts=""
```
Expected: fail before implementation.

- [ ] **Step 2: Implement caveat application**

At claim-builder/section-composer boundary, apply `claim_review_suggestions` by changing section writing mode and strength, not by deleting unrelated evidence.

- [ ] **Step 3: Verify**

Run:
```powershell
python -m pytest tests\test_claim_depth_pack.py tests\test_report_contracts_and_composer.py -q -o addopts=""
```
Expected: pass.

### Task 5: Make Search Task Drops Diagnostic Unless They Remove All Evidence

**Files:**
- Modify: `rag_pipeline/agents/brain_agent.py`
- Modify: `rag_pipeline/flows/report/full_report.py`
- Test: `tests/test_search_task_topic_anchor_contract.py`
- Test: `tests/test_report_quality_regressions.py`

- [ ] **Step 1: Write failing tests**

If `retrieval_dropped_count > 0` but each final chapter has claim-backed evidence, assert score has warning only, not clean blocker.

- [ ] **Step 2: Implement**

Map `search_tasks_dropped` to warning unless `analysis_ready_fact_count == 0` or final claim coverage is zero.

- [ ] **Step 3: Verify**

Run:
```powershell
python -m pytest tests\test_search_task_topic_anchor_contract.py tests\test_report_quality_regressions.py -q -o addopts=""
```
Expected: pass.

### Task 6: Replay Validation

**Files:**
- No source changes; validation only.

- [ ] **Step 1: Run targeted tests**

```powershell
python -m pytest tests\test_citation_manifest.py tests\test_final_audit_agent.py tests\test_report_quality_regressions.py tests\test_report_pipeline_replay.py -q -o addopts=""
```

- [ ] **Step 2: Run full tests**

```powershell
python -m pytest -q -o addopts=""
```

- [ ] **Step 3: Run deterministic replay**

```powershell
$env:REPORT_BLUEPRINT_SOURCE='claim_first'
$env:REPORT_COMPOSER_EXPAND_TO_TARGET='true'
$env:REPORT_COMPOSER_TARGET_SECTION_CHARS='520'
$env:REPORT_ENABLE_RENDERER_TEMPLATE_EXPANSION='true'
$env:REPORT_RENDER_MIN_SECTION_CHARS='420'
python scripts\replay_stage.py --run-id "20260623_153232_中国低空经济产业链商业化机会与风险分析_2026" --from analysis --output-dir output\publishability_repair_replay_YYYYMMDD_HHMMSS
```

Success criteria:
- `body_char_count >= 6500`
- `final_body_char_count >= 6000`
- no internal markers in body
- FinalAudit has no fatal delivery blockers
- evidence quality findings remain warnings
- citation mismatch findings decrease or disappear
