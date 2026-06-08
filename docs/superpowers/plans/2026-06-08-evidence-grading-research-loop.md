# Evidence Grading And Research Loop Optimization Plan

## Goal

把当前“证据审查过严导致分析/写作被压扁”的链路，改成分层证据策略：

- 分析层更宽：允许方向性、案例性、线索性证据进入分析，但必须带边界。
- 写作层受控：只能按证据等级生成不同强度的表达，不能把弱证据写成强结论。
- 终审层保硬：错引用、无引用事实、伪来源、内部标记仍然 fatal。
- 补正层更准：不足时输出 gap-driven search task，而不是让模型重吞全文自由补写。

## Current Diagnosis

真实联网 run 暴露出的主要问题不是“完全没有证据”，而是：

1. `FinalAudit` 把 citationless factual body 和 final citation gap 混在一起，导致同一个根因重复 fatal，补正定位不干净。
2. `analysis_agent.validate_llm_analysis_output` 对 incomplete metric 直接 `continue` 丢 claim；这会把可作为方向性判断的线索也杀掉。
3. 证据门当前偏全局硬门，缺少按 `proof_role / claim_strength / evidence_use_level` 的分级使用。
4. Doubao 类 deep research 的优势是“搜索后反思再补证”，而当前 RAG2 已有 gap/repair 账本雏形，但还没形成 research memo 到 repair task 的闭环。
5. 评分和阻塞策略需要分离：结构分不足可以降交付等级，但不能和事实错误使用同一种 fatal 语义。

## Phases

### P0: Clean Deterministic Diagnostics

Scope:

- 修复 final citation audit 的重复误报。
- 只有正文引用和来源附录确实不一致时才产 `final_citation_gap`。
- citationless factual body 单独保留为 fatal。

Acceptance:

- 当 `final_body_citation_refs == final_appendix_refs` 且 `final_missing_appendix_refs=[]` 时，不因为 reconciliation status 是 `blocked` 而额外产 `final_citation_gap`。
- 仍然保留 `citationless_factual_body`。

### P1: Evidence Grading In Analysis

Scope:

- 对 numeric claim 引用了 incomplete metric card 的情况，不直接丢弃 claim。
- 将该 claim 降为 `directional`，设置：
  - `evidence_use_level=directional_signal`
  - `writing_permission=cautious_with_boundary`
  - `metric_completeness_status=incomplete`
  - `metric_missing_fields`
  - `limitation_boundary` 追加缺口说明
- 仍然拒绝明显 unsupported 的实体/数字错配，避免放宽成错引用。

Acceptance:

- incomplete metric claim 保留为 usable claim，但不能是 strong/moderate。
- unsupported entity/number claim 仍然被拒绝。
- issue 计数仍记录 `llm_numeric_claim_incomplete_metric_fact`，用于后续 repair。

### P2: Research Reflection Memo

Scope:

- 在 evidence merge / analysis input 附近生成 compact research reflection memo：
  - known_findings
  - coverage_by_requirement
  - ambiguous_or_weak_claims
  - missing_fields
  - next_search_task_seeds
  - enough_to_write
  - write_mode
- memo 只给 analysis/repair，不直接给 writer 当事实来源。

Acceptance:

- memo 能从 `score_gaps` / requirement gaps / evidence quality 中生成。
- `enough_to_write=false` 时 writer 只能产 limited draft 或 short honest draft。

### P3: Gap-Driven Repair Search Seeds

Scope:

- `repair_context_view` 的 `repair_task_seed` 必须带 `requirement_id/gap_id/proof_role/required_fields/reject_if/success_criteria`。
- 对 `still_insufficient/live_search_required` 自动生成更精确 query seed。
- 避免重复 failed query。

Acceptance:

- 同一 gap 的下一轮 query 包含缺失字段和目标来源类型。
- repair view 不返回正文可引用事实、旧 section、raw page。

### P4: Calibrated Quality Gate

Scope:

- 保持事实错误、错引用、伪来源、内部标记为 fatal。
- 将结构不足（正文短、表格少、search tasks dropped）改成 delivery tier / score 降级，不直接等同 fatal。
- A/B 来源要求按 proof_role 分层：metric/filing/核心结论严，case/context/counter 可以 directional。

Acceptance:

- final status 区分 `publishable`, `limited_review_draft`, `short_honest_draft`, `blocked_fact_integrity`。
- 结构不足可触发补正，但不污染 citation integrity 判断。

### P5: Live Evaluation

Scope:

- 使用同一真实 query 跑带联网、真实模型、全开配置的对照。
- 记录：
  - usable_claim_count
  - final_audit fatal types
  - citationless count
  - final_citation_gap count
  - qa_quality_score
  - body chars
  - A/B source usage
  - token/cost by stage

Acceptance:

- P0/P1 后，claim 不再因 incomplete metric 被整批压掉。
- FinalAudit gap 类型更准。
- 如果仍不 publishable，能明确是 evidence gap、body length、table data，还是 citation integrity。

## Implementation Order

1. Add tests for P0/P1.
2. Implement P0 in `rag_pipeline/flows/report/final_audit_agent.py`.
3. Implement P1 in `rag_pipeline/agents/analysis_agent.py`.
4. Run targeted tests.
5. If green, continue P2/P3 in context/repair modules.
6. Run full tests and one live evaluation.

## Guardrails

- 不降低错引用、伪来源、无引用事实的终审要求。
- 不让 weak/directional 证据生成 strong claim。
- 不把 repair/diagnostic/memo 内容直接送入 writer 当事实。
- 所有新增 prompt/contract 行为必须有 schema/test 锁住。
