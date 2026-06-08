# RAG2 补正逻辑优化计划（Gap-Driven Repair Spine）

> 目标：把今天分散在 4 条循环里、各自算 gap/各自补的补正流程，整合成**单一 gap 账本驱动 + 类型隔离 + 共享去重**的 repair spine。
> 状态：依赖缓存计划 [CACHE_OPTIMIZATION_PLAN.md](CACHE_OPTIMIZATION_PLAN.md) 的 Phase 1（requirement_id 脊柱）、Phase 3（artifact_class 守卫）、Phase 4（context view）；本文档假设三者已就位。
> 日期：2026-06-04。状态：设计中。

---

## 0. 第一原则（继承 cache plan §0）

加性优先 / flag 默认=现状 / fail-open / 先 warn 后 enforce / 每阶段冒烟回归 / 不换引擎。

新增一条**补正专属契约**：
- **N1 单一 gap 账本**：所有循环只看一个 `evidence_gap_ledger`，不再各自算 gap；老的 gap 重算逻辑保留为"对账校验"，发现不一致只报警不接管。
- **N2 不裁切并行**：4 条循环可以共享账本但**不合并成 1 条**——layout 和 post_qa 解决的问题不同（一个补证据、一个改结构）。统一的只是 gap 来源和去重，不是执行体。

---

## 1. 现状诊断（实代码）

**4 条循环 + 1 个 in-memory 账本 + 1 个 repair_task DTO**

| 循环 | 入口 | 上限 | 是否调 LLM | 输出 |
|---|---|---|---|---|
| supervisor coverage | `run_supervisor_evidence_loop` brain_agent.py:10859 | env 5（硬上限 8） | ✓ `evaluate_coverage_with_llm` L5288 | follow_up_queries → `_repair_tasks_from_items`（seen_keys A） |
| layout refinement | layout_claim_matcher / L9417 | env 3（硬上限 6） | 规则为主 | layout 调整 |
| post_qa_repair | `_run_post_qa_repair_round` L9024（调用 L9470 + L9673 共 2 次） | 各 1 次 | 部分（reformatter+repair_plan） | 重跑章节/修复 |
| writer rewrite | rewrite_agent.py | — | ✓ 但默认关（`REPORT_ENABLE_LLM_REWRITE` L213 默认 False） | 改正文 |

**已有雏形（别重建）**：
- `evidence_gap_ledger`（analysis_agent.py:3250 构建、brain_agent.py:3325/3393 流转、3783 handoff）——结构化 gap 列表
- `_repair_task_from_item`（brain_agent.py:8393）——已归一化 repair_task DTO，含 `gap_id/hypothesis_id/blocking_gaps/proof_role/lane_targets/required_fields/agent`，metric 类自动填 required_fields
- `_repair_tasks_from_items`（L8427）——批量 + seen_keys 去重，**已被 5 处循环调用**（L8936/9056/9422/9609/11010）
- `build_followup_queries_for_chapter`（L4795）——规则化：按 `missing_total_sources/missing_metric_scope_period_unit/missing_counter` 标签生成查询，无 LLM
- `gap_repair_round` 计数（L7733）——单 task 已有轮次记录

**真正的缺口**：
- **D-1 gap 账本不持久、不可查询**：evidence_gap_ledger 只活在 evidence_package dict 里，跨循环靠 dict 传递，不在 SQLite/snapshot 里以 requirement_id 可索引方式存在。
- **D-2 4 条循环各自 seen_keys**：`_repair_tasks_from_items` 5 处调用各自构造 `seen_keys: set()`。supervisor 已查过的 gap，post_qa 可能又查一次。
- **D-3 缺 repair_context_view**：LLM coverage eval（evaluate_coverage_with_llm L5304）传 `evidence_pool_summary` + `coverage_units`，已较瘦；但 post_qa_repair 路径没有类型守卫，可能让 LLM 看旧 section → 引入新事实风险。
- **D-4 sufficiency_check 隐式**：补完一轮，是否达标靠"再次 coverage eval"或"规则计数"判断，没有以 requirement_id 为单位的显式 sufficiency 表，无法局部判停。
- **D-5 close_gap 没显式状态机**：gap 状态隐藏在 ledger entry 字段里（gap_type/severity），缺 `pending/probing/satisfied/abandoned` 的明确状态流转。

---

## 2. 目标架构（Gap-Driven Repair Spine）

```
                ┌───────────────────────────────────────┐
                │ evidence_gap_ledger（持久化到 SQLite）  │  ←— 唯一 gap 来源
                │  + requirement_id + state + provenance │     依赖 cache Phase 1
                └─────────────────┬─────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │  GapStateMachine（pending→probing → │
                │  satisfied / abandoned / blocked） │
                └─────────────────┬─────────────────┘
                                  │
       ┌──────────────────────────┼─────────────────────────┐
       ▼                          ▼                         ▼
   repair_dispatcher        sufficiency_check        repair_context_view
   （选哪个 loop 处理）      （以 req_id 为单位判停）  （白名单切片，依 cache Phase 4）
       │
       ├─→ ① supervisor (evidence gap)
       ├─→ ② layout    (structure gap)
       ├─→ ③ post_qa   (citation/coverage gap)
       └─→ ④ rewrite   (prose gap, 默认关)
   ↑ 4 条循环都从 ledger 取 task、都写 ledger 回执，共享去重；不再各算 gap
```

核心边界（写入硬校验门 G6，扩展 cache plan §3.3）：
- 任何 loop 的 LLM prompt 必须经 `repair_context_view`，禁止直接拿 raw section/raw search snippet/raw page。
- LLM 在补正阶段**只允许输出 repair_task DTO，不允许输出 fact_card 或 prose**。
- post_qa 和 writer rewrite 引入新数字/公司名/日期时（cache plan G5 lint pass），强制走 fact_card 校验。

---

## 3. 分阶段执行计划

### Phase R0 — gap 账本持久化（依赖 cache Phase 1）｜~1 天

**目标**：让 evidence_gap_ledger 在 SQLite 里可按 `(run_id, requirement_id, state)` 查询。

**改动点**
- 沿用 cache Phase 1 的 lineage envelope，新增 SQLite 表（在 evidence_cache.sqlite 里，不另起库）：
  ```sql
  CREATE TABLE IF NOT EXISTS gap_ledger (
    gap_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    requirement_id TEXT,
    chapter_id TEXT,
    hypothesis_id TEXT,
    gap_type TEXT NOT NULL,     -- missing_total_sources / missing_metric / missing_counter / citation_gap / structure_gap
    severity TEXT,
    state TEXT NOT NULL,        -- pending / probing / satisfied / abandoned / blocked
    blocking_gaps_json TEXT,
    required_fields_json TEXT,
    proof_role TEXT,
    discovered_in TEXT,         -- supervisor / layout / post_qa / writer
    discovered_round INTEGER,
    last_repair_round INTEGER,
    last_task_id TEXT,
    payload_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_gap_run_req ON gap_ledger(run_id, requirement_id, state);
  CREATE INDEX IF NOT EXISTS idx_gap_run_state ON gap_ledger(run_id, state);
  ```
- 新增 `rag_pipeline/cache/gap_ledger.py`：`upsert_gap / fetch_open_gaps / set_state / record_attempt`。
- `analysis_agent.py:3284` 构建 `evidence_gap_ledger` 时 + brain_agent.py 流转处，附带调用 `upsert_gap`（fail-open）。

**新增 flag**：`GAP_LEDGER_PERSIST_ENABLED`（默认 false → 验证后 true）。

**验证**：跑 golden case → `SELECT count(*) FROM gap_ledger WHERE run_id=?` >0；`cache_report` 加 gap_ledger section。

**回滚**：flag 关。

---

### Phase R1 — 共享 seen_keys（消除 D-2）｜~0.5 天

**目标**：4 条循环共享同一个去重集合，同一 gap 不会被反复发现。

**改动点**
- `_repair_tasks_from_items`（brain_agent.py:8427）支持传入 `seen_keys` 已有 5 处调用——把 5 处调用统一改为从 `state["repair_seen_keys"]` 取同一个 set，run 级初始化。
- 增 `repair_attempt_log`（gap_ledger 子表或 jsonl），记录 `(gap_id, loop_name, round, task_key, outcome)`，便于诊断同 gap 多次尝试。

**新增 flag**：`REPAIR_SHARED_DEDUP_ENABLED`（默认 false）。

**验证**：开 flag 跑 golden → `attempt_log` 中无同 `(gap_id, task_key)` 重复条目；总 followup 调用数下降，正文质量持平。

**回滚**：flag 关，回到各自 set。

---

### Phase R2 — repair_context_view（依赖 cache Phase 4）｜~1.5 天

**目标**：所有补正 LLM 调用走白名单视图，杜绝旧 section + 新事实污染。

**改动点**
- 在 cache Phase 4 的 `rag_pipeline/context/context_view_builder.py` 增第三视图：
  ```python
  def repair_context_view(*, run_id, gap_id) -> Dict:
      return {
          "gap": {gap_id, requirement_id, gap_type, severity, required_fields, blocking_gaps},
          "hypothesis": ...,
          "existing_evidence_summary": ...,        # 来自 ledger 查询，不是 raw section
          "previous_attempts": [{round, query_key, outcome}, ...],
          "instruction": "只输出 repair_task DTO；禁止输出 fact_card/prose/数字/公司名/日期。",
          "forbidden": ["raw_section", "raw_search_snippet", "raw_page",
                        "rejected_facts", "stale_facts", "old_section_draft"],
          "schema": REPAIR_TASK_DTO_SCHEMA,
      }
  ```
- 4 条循环的 LLM 入口替换：
  - `evaluate_coverage_with_llm`（brain_agent.py:5288）→ 包一层 view。
  - `_run_post_qa_repair_round`（L9024）走 view。
  - rewrite_agent.py 的 LLM 调用走 view（开启时）。
  - layout 循环无 LLM，跳过。
- 输出层加 schema 校验：LLM 返回的 JSON 不符合 `REPAIR_TASK_DTO_SCHEMA` → 降级走规则化 `build_followup_queries_for_chapter`。

**新增 flag**：`REPAIR_CONTEXT_VIEW_ENABLED`（默认 false）。

**验证**：开 flag 跑 golden → telemetry/context_budget.py 测得各 LLM 调用 token 数显著下降；正文 fact 数 ≥ 基线。

**回滚**：flag 关。

---

### Phase R3 — GapStateMachine + sufficiency_check（解 D-4/D-5）｜~2 天

**目标**：以 requirement_id 为单位显式判停，不再每轮全 coverage 重算。

**改动点**
- gap_ledger 状态机：`pending → probing → satisfied | abandoned | blocked`。
  - `probing`：已派 task 等结果。
  - `satisfied`：新证据 + sufficiency 校验通过。
  - `abandoned`：超过 `gap_repair_round` 上限（沿用 L7733 现有计数，默认 3）。
  - `blocked`：依赖另一未关闭 gap（blocking_gaps 非空且未全 satisfied）。
- `sufficiency_check(requirement_id, run_id) -> {state, missing_fields, ok}`：
  - 输入：该 requirement 当前 fact_cards（cache Phase 1 索引读路径 `lookup_evidence_by_requirement`）。
  - 检查：required_fields 是否齐全 + min_source_level 是否满足 + 是否多源（≥2 来源验证）。
  - 输出驱动 state 流转。
- 每条循环开局先 `fetch_open_gaps(run_id)` 拉待办，循环结束 `record_attempt` 写回。

**新增 flag**：`REPAIR_STATE_MACHINE_ENABLED`（默认 false）。

**验证**：跑 golden → 每个 gap 在 ledger 里能追到完整状态轨迹；总 LLM coverage eval 调用数下降（同 requirement 已 satisfied 不再评估）。

**回滚**：flag 关，回到隐式判停。

---

### Phase R4 — repair_dispatcher（解 D-1/D-2 总体）｜~2 天

**目标**：4 条循环对外是 4 个 worker，对内由 dispatcher 从单一账本派发任务，避免重叠。

**改动点**
- 新增 `rag_pipeline/agents/repair_dispatcher.py`：
  ```python
  def dispatch_open_gaps(run_id) -> Dict[str, List[Task]]:
      gaps = fetch_open_gaps(run_id)
      by_kind = {"evidence": [], "structure": [], "citation": [], "prose": []}
      for g in gaps:
          if g.gap_type in EVIDENCE_GAP_TYPES: by_kind["evidence"].append(g)
          elif g.gap_type in STRUCTURE_GAP_TYPES: by_kind["structure"].append(g)
          ...
      return by_kind
  ```
- supervisor / layout / post_qa / rewrite 启动前调 dispatcher 拿"属于自己的 task"。
- 每条循环关闭时调 `set_state(gap_id, state)`，下一条循环看到的就是更新后的账本。

**新增 flag**：`REPAIR_DISPATCHER_ENABLED`（默认 false）。

**验证**：开 flag → 跑 golden，对比 cache_report.repair_log：
  - 同 gap_id 不会在 supervisor 和 post_qa 各 attempt 一次；
  - 总 attempt 数下降 ≥20%（用户那句"成本下降"在这里兑现）。

**回滚**：flag 关。

---

### Phase R5 — 硬校验门 G6（依赖 cache Phase 3）｜~0.5 天

**目标**：补正阶段的事实/派生隔离，落到守卫。

**改动点**
- 扩展 cache plan §3.3 的 G4：`artifact_class=repair_diagnostic` 标签独立，绝不进 fact_card 写入路径。
- `_run_post_qa_repair_round` 任何 LLM 输出经 lint：含未在 fact_card 中出现的数字/公司名/日期 → 拒绝写入。
- rewrite_agent 启用时，输出强制走 fact_card 校验。

**新增 flag**：随 `CACHE_VALIDATION_MODE` 联动（off/warn/enforce）。

**验证**：构造一个故意带"未授权数字"的 LLM mock 输出 → enforce 模式下被拒。

**回滚**：MODE=off。

---

## 4. 与 cache plan 的依赖图

```
cache Phase 1 (requirement_id 脊柱) ─────┬─→ R0 (gap 持久化)
                                          ├─→ R3 (sufficiency_check 用 lookup_by_requirement)
cache Phase 3 (G4 artifact_class)  ──────┼─→ R5 (类型隔离)
cache Phase 4 (context view 框架)  ──────┴─→ R2 (repair_context_view)
```

执行顺序：cache 0/1 → R0/R1 → cache 2/3 → R5 → cache 4 → R2/R3 → R4。
不强制按顺序，但 **R0 必须在 cache 1 之后**，**R2 必须在 cache 4 之后**，**R5 必须在 cache 3 之后**。

---

## 5. 预期收益（按用户提问对账）

| 你的期待 | 兑现于 |
|---|---|
| "缓存之前数据" | cache Phase 1（脊柱）+ R0（gap 持久化） |
| "LLM 只返回补正方向" | R2 repair_context_view + schema 校验 |
| "局部重跑不全跑" | cache Phase 5 + R3 sufficiency_check |
| "成本下降" | R1 共享 seen_keys + R4 dispatcher，少跑重复 task |
| "质量更稳" | R2 上下文白名单 + R5 类型隔离 + cache G3 ceiling 强制 |
| "repair view 不给可引用事实" | R2 forbidden 字段 + R5 G6 守卫 |

---

## 6. 风险

| 风险 | 缓解 |
|---|---|
| 状态机引入死锁（blocking_gaps 循环依赖） | R3 增 cycle detection，cycle 直接 abandon |
| dispatcher 把 task 派给错的 loop | R4 先 warn-mode 跑 N 次 golden，看分类准确率 |
| 持久化 gap 跨 run 复用导致旧 gap 复活 | gap_ledger 永远按 run_id 隔离；跨 run 复用走 fact_card 而非 gap |
| writer rewrite 默认关导致 R5 验证不充分 | 评估时显式开 `REPORT_ENABLE_LLM_REWRITE=true` 跑专项验证 |

---

## 7. 不做什么（明确边界）

- **不合并 4 条循环为 1 条**——它们解决的问题域不同。统一的是 gap 来源、去重、判停，不是执行体。
- **不重写 `_repair_task_from_item`**——已有 DTO 雏形足够，只补 `state/discovered_in/last_repair_round` 字段。
- **不动 LLM coverage eval 的 prompt**——它今天已经只看 summary 不看原文，问题不在它。
- **不引入新数据库**——gap_ledger 进 evidence_cache.sqlite 同库，沿用 cache plan 的"不换引擎"原则。

---

## 8. 落地顺序速查

cache Phase 0 ✅ → cache Phase 1 → **R0** → **R1** → cache Phase 2 → cache Phase 3 → **R5** → cache Phase 4 → **R2** → **R3** → **R4** → cache Phase 5。

最先动手且 ROI 最高的是 **R0 + R1**（gap 持久化 + 共享去重），它们能马上把"同一 gap 被多 loop 重复发现"的浪费砍掉，不依赖 cache Phase 3/4。
