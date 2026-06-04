# RAG2 缓存优化计划（Artifact Ledger 化）

> 目标：把现有零散的 6 个存储 + 契约层，升级为以 `run_id + requirement_id` 为脊柱的可追溯、可局部重跑、可校验的缓存体系。
> 合成自：用户的「三类/四层缓存 + ledger + context view」方案 + 实代码诊断（见 `docs` 同目录或会话记忆 `rag2-cache-stack-reality`）。
> 日期：2026-06-04。状态：Phase 0 已实现并静态验证；待捕获首个基线后进入 Phase 1。

---

## 0. 第一原则：不破坏现有链路（「全流程跑通」的硬契约）

每一阶段都必须遵守，违反则该阶段作废重做：

1. **加性优先**：只新增列/字段/函数/模块；不删列、不改键、不迁移已有数据。SQLite 用 `ALTER TABLE ADD COLUMN`（可空），JSONL 天然 schemaless。
2. **flag 门控，默认=现状**：每个行为变更挂 env flag，默认值保持当前行为；全部新 flag 关闭时，管线输出必须与基线**逐字节/逐字段一致**。
3. **fail-open**：新代码路径任何异常都回退到现有路径，绝不让缓存层抛错中断报告交付（沿用现有 `_safe()`、`write_stage_snapshot_safe` 风格）。
4. **先观察后强制**：校验门先以 `warn`（只记日志/计数，不拦截）跑若干 golden case，看违规率，再切 `enforce`。
5. **每阶段冒烟回归**：改完跑 §8 的冒烟，diff `cache_report` 与正文，确认无回归再进下一阶段。
6. **不换引擎**：本计划全程不引入 Postgres/Redis/OSS。引擎迁移是独立的 Phase 6，触发条件见 §9，当前不做。

---

## 1. 现状一页纸（诊断结论）

**6 个存储（键各不相同，无一以 requirement_id 为键）：**

| 存储 | 后端 | 写入身份键 | 读取命中 | freshness |
|---|---|---|---|---|
| evidence_cache.search_cache | SQLite | `search:`+sha256(归一query+搜索参数) | 精确哈希 | TTL 按源类型+负缓存6h |
| evidence_cache.evidence_cache | SQLite | `ev:`+sha256(url+fact+metric+period+value) | 模糊(300行→Python过滤) | TTL 按源类型 |
| evidence_cache.evidence_lineage | SQLite | autoincrement | 按 evidence_id | — |
| topic_bundle | 文件 | `slug(query)__hash(query+family+geo+time)` | 别名扫描+schema/污染/age门 | max_age_days=30 |
| trusted_source | JSONL | `trusted:`+sha256(url+title+fact+metric+period+value) | term 模糊 | last_verified_at |
| stage_snapshot | 文件 | `run_id+stage` | 按 run_id+stage | 无 |
| runtime_cache | 内存 | `ns:`+sha256(payload) | 精确键+TTL | 重启失 |

**契约层（纯函数校验器，非存储）：** `evidence_quality.py`(A/B/C/D裁决器)、`evidence_ledger.py`(内存 lineage 变换)、`quality_gate.py`、`report_contract.py`、`source_registry.py`(引用renumber工具)。

**核心缺口（本计划要解决的）：**
- **G-1 无统一键**：`requirement_id` 只活在内存 fact_card（`analysis_agent.py:2045`），落任何持久层全部蒸发（`grep requirement_id rag_pipeline/cache/` = 0）。→ 按 requirement_id 局部重跑做不到。
- **G-2 lineage 碎片化**：无任何持久记录同时握有 `requirement_id + source_id/fact_id + run_id`。
- **G-3 身份/裁决不统一**：evidence_cache 收 C 级、trusted_source 只收 B+；两套 content-hash（trusted 多了 title）；source_level 三处各算 → 跨库无法去重/对账，判定可能矛盾。
- **G-4 事实/派生无守卫**：stage_snapshot 把 `evidence_package`(事实) 与 `writer_report/qa_result`(派生) 同机制存，无类型标记阻止派生被当证据读。
- **G-5 无跨阶段白名单 context view**：`search/context_builder.py` 是检索召回，不是给 analysis/writer 的白名单切片。
- **G-6 query-intent freshness 缺**：TTL 按 source_type 给（新闻 2 天），对「今日/最新」类查询偏长。

**已有、不要重建的（重要）：** content-hash 去重身份、TTL 按源分级、负缓存、命中遥测、topic 跨run复用、污染/schema门、A/B/C/D+子级契约、claim_strength_ceiling（`brain_agent.py:1068` 计算、`analysis_agent.py:2191` 挂载）。

---

## 2. 目标架构（在已有存储之上加薄脊柱，不拆不换）

```
                      ┌─────────────────────────────────────────────┐
                      │  Lineage Envelope（数据契约，§5）              │
                      │  {run_id, requirement_id, source_id, fact_id, │
                      │   search_task_id, chapter_id, hypothesis_id,  │
                      │   claim_id, proof_role}                       │
                      └───────────────┬─────────────────────────────┘
   planning(brain/planner) → search_task → fact_card(analysis_agent) ── 顺流携带，落库不丢
                                      │
   ┌──────────────┬──────────────────┼───────────────────┬──────────────────┐
   ▼              ▼                   ▼                    ▼                  ▼
 search_cache  evidence_cache   trusted_source        topic_bundle      stage_snapshot
 (+freshness)  (+req_id列,+ver) (+req_id字段,统一身份)  (+ver字段)        (+artifact_class,+ver)
   └──────────────┴──────────────────┴───────────────────┴──────────────────┘
                                      │
                      ┌───────────────┴──────────────┐
                      │  cache_registry（门面/索引，§3.2）│  ← 唯一对外读写入口，知道所有存储
                      └───────────────┬──────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
 context_view_builder(§3.5)    hard validation gates(§3.4)    local rerun driver(§3.6)
 analysis/writer/repair 白名单   G1..G5                          按 requirement_id 重跑
```

三件核心增量（全部 storage-agnostic，迁 Postgres 不用重做）：
1. **贯穿 requirement_id + run_id** 到所有持久落点（解 G-1/G-2）。
2. **统一 evidence 身份 + 单一 source_level 裁决**（解 G-3）。
3. **硬校验门 + artifact_class 类型守卫 + 白名单 context view**（解 G-4/G-5，这是质量保护，不是降本）。

---

## 3. 分阶段执行计划

> 每阶段格式：目标 / 改动点(file) / 新增 schema·flag / 验证 / 回滚。
> 估时为单人投入的粗估，仅排序参考。

### Phase 0 — 验证护栏与可观测（不改任何行为）｜~0.5 天 ✅ 已落地（2026-06-04）

**目标**：建立基线，使后续每阶段都能 diff 证明「无回归」。

**已实现（静态验证通过：4 文件编译 OK + build_cache_report fail-open 冒烟 OK）**：
- 新增 `rag_pipeline/cache/cache_report.py`：`build_cache_report/write_cache_report`，聚合 evidence_cache(activity+stats)/stage_snapshots/trusted_source/topic_bundle，每源各自 try/except。
- 新增 `rag_pipeline/cache/trusted_source_cache.py::trusted_source_stats()`（只读统计）。
- `flows/report/full_report.py`：L4592 后插入 fail-open sidecar 写入块（`env_flag("CACHE_REPORT_SIDECAR_ENABLED", True)`），**只写 `<base_name>.cache_report.json`，不改任何既有 payload**。
- 新增 `tools/cache_baseline.py`：跑 golden query 端到端并收割 报告+cache_report+摘要。
- 捕获基线命令：`python tools/cache_baseline.py --golden first --label before_phase1`（需网络/LLM，耗时数分钟）。关闭观测：`CACHE_REPORT_SIDECAR_ENABLED=0`。

**改动点**
- `flows/report/full_report.py`：报告产出 metadata 末尾，汇总一个 `cache_report` blob（聚合现有 `evidence_cache_activity_summary()`、`topic_bundle_seed_evidence_count`、各 stage_snapshot manifest、trusted_source 命中）。只读聚合，不改逻辑。
- 新增 `tools/cache_baseline.py`：跑一个 golden query → 落 `output/cache_baseline/<ts>/`（正文 md + cache_report.json + 关键 snapshot manifest 列表）。

**新增**：无 schema、无 flag（纯新增工具 + 只读聚合）。

**验证**：`python run_full_report.py "<golden query>" --route web --output-dir output/_baseline`；产出 `cache_report` 字段齐全；正文与改动前一致。

**回滚**：删工具文件即可。

---

### Phase 1 — requirement_id 脊柱（贯穿落库 + 索引）｜~2 天｜**最高优先**

**目标**：requirement_id/run_id 落到所有持久层，并提供按 requirement_id 的索引读路径。解 G-1/G-2。

**改动点**
- 新增 `rag_pipeline/cache/lineage.py`：定义 `LineageEnvelope`（dataclass + `to_dict`/`from_item`），字段见 §5。来源 = 内存 fact_card（`analysis_agent.py:2045` 已有 requirement_id/fact_id/source_id）+ search_task（已有 requirement_id）+ run 级 run_id/report_id。
- `cache/evidence_cache.py`：
  - `_ensure_schema()`(L385)：追加 `ALTER TABLE evidence_cache ADD COLUMN requirement_id TEXT`；`ALTER TABLE evidence_lineage ADD COLUMN requirement_id TEXT / chapter_id TEXT / claim_id TEXT / hypothesis_id TEXT`。用 try/except 包裹（旧库已存在列时静默跳过）；`PRAGMA user_version` 升到 2。新增 `CREATE INDEX idx_evidence_requirement ON evidence_cache(requirement_id)`。
  - `_record_from_evidence()`(L917) / `store_evidence_from_package()`(L796)：从 item 携带 requirement_id 落 `requirement_id` 列；lineage insert 补 requirement_id/chapter_id/claim_id/hypothesis_id。
  - 新增 `lookup_evidence_by_requirement(requirement_id, *, min_source_level=...)`：走 `idx_evidence_requirement` 精确索引（不替换现有模糊 `lookup_evidence`，是新增的局部重跑专用读路径）。
- `cache/trusted_source_cache.py`：`_entry_from_evidence()`(L208) 已收 run_id/report_id，追加 `requirement_id` 到 entry dict（JSONL 直接加键，零迁移）。
- 取证写入侧（`agents/evidence_merger.py`、`agents/web_analysis_agent.py`）：把 envelope 透传到 store 调用（不改取证逻辑，只把已有的 requirement_id 往下递）。

**新增 flag**
- `CACHE_REQUIREMENT_SPINE_WRITE`（默认 **true**）：落 requirement_id 列/字段。加性写，安全默认开。
- `CACHE_REQUIREMENT_LOOKUP_ENABLED`（默认 **false**）：启用 `lookup_evidence_by_requirement` 读路径。

**验证**：跑 golden case → `sqlite3` 查 `SELECT count(*) FROM evidence_cache WHERE requirement_id IS NOT NULL`（应 >0）；trusted JSONL 出现 requirement_id；正文与基线一致。

**回滚**：`CACHE_REQUIREMENT_SPINE_WRITE=false` 即停止写新列；列本身可空，留着无害。

---

### Phase 2 — 统一身份与单一裁决（消除 G-3）｜~2 天

**目标**：evidence 跨库可去重对账；source_level 单一权威。

**改动点**
- `cache/lineage.py` 增 `evidence_identity(item) -> str`：单一 content-hash 配方（url + 归一fact + metric + period + value；**title 不入 hash** 以兼容 evidence_cache 现配方）。evidence_cache 的 `ev:` 与 trusted_source 的 `trusted:` 改为复用它（保留旧 id 双读一个过渡期：先按新 id 查，miss 再按旧 id 查并回填）。
- `contracts/evidence_quality.py`：`classify_evidence` 成为唯一 source_level 裁决；`evidence_cache._source_level`(L184) 与 `trusted_source_cache._source_level`(L145) 改为委托它。trusted_source 的「只收 B+」改为显式策略：`classify` 出级别后按 `TRUSTED_SOURCE_MIN_LEVEL`(默认 B) 过滤，使「同一来源两库判定一致」。

**新增 flag**
- `CACHE_UNIFIED_IDENTITY_ENABLED`（默认 **false** → 验证后切 true，含双读过渡）。
- `EVIDENCE_QUALITY_SINGLE_AUTHORITY`（默认 **false** → parity 比对通过后切 true）。

**验证**：构造同一来源样本喂两库 → 断言 evidence_identity 相同、source_level 相同；跑 golden case，cache_report 的 evidence 去重数应较基线下降或持平（不应上升）。

**回滚**：两 flag 关 → 回到各自旧身份/旧裁决。

---

### Phase 3 — 硬校验门 + 类型守卫（解 G-4，质量保护）｜~3 天

**目标**：防三类错误——引用不存在的 fact、弱证据写成强结论、派生当事实。**这是质量保护，不是降本。**

**门（统一一个 `CACHE_VALIDATION_MODE` = off|warn|enforce，默认 off → 灰度 warn → enforce）**
- **G1 引用存在性**：section/claim 引用的 fact_id/source_ref 必须存在于本 run 的 evidence/source_registry。落点：`agents/final_writer_agent.py` 渲染前 + `agents/qa_agent.py`。
- **G2 可采性**：被引用证据必须 validated/admissible，不得是 rejected/stale/superseded。落点：同 G1，复用 `evidence_quality` 状态。
- **G3 claim_strength ≤ ceiling**：强制已存在的 `claim_strength_ceiling`（`analysis_agent.py:2191` 已挂载，但当前未见强制）。超限则降级 claim_strength 到 ceiling 并记 diagnostic。落点：`agents/claim_builder_agent.py` / `agents/chapter_argument_agent.py`。
- **G4 事实/派生隔离**：`cache/stage_snapshot_cache.py` 的 manifest 增 `artifact_class`（fact|derived|diagnostic，按 stage 映射：evidence_package→fact，writer_report/qa_result→derived，gap 类→diagnostic）；任何「证据加载」路径加守卫：拒绝从 `artifact_class!=fact` 的 snapshot 取证据。
- **G5 正文无新事实**：final prose 不得出现未在 claim_unit 中的数字/公司名/日期。落点：`agents/qa_agent.py` 增一个 lint pass（warn 起步）。

**新增 flag**：`CACHE_VALIDATION_MODE`（默认 off）；各门可单独 `CACHE_GATE_G{1..5}`（默认随 MODE）。

**验证**：MODE=warn 跑 golden case → 看 `cache_report.validation` 各门违规计数；人工核违规是否真问题；再 MODE=enforce 跑，确认报告仍产出（fail-open：enforce 命中时降级而非崩溃）。

**回滚**：`CACHE_VALIDATION_MODE=off`。

---

### Phase 4 — context_view_builder（白名单切片，解 G-5）｜~2 天

**目标**：模型只看当前 requirement 的干净切片，不直接读脏缓存/数据库。

**改动点**
- 新增包 `rag_pipeline/context/`（当前不存在；`search/context_builder.py` 是检索召回，不复用）：`context_view_builder.py`，三视图：
  - `analysis_context_view(requirement_id)` → 仅 `{requirement_id, hypothesis, usable_fact_cards[], claim_units[], missing[], forbidden:[rejected,stale,superseded], instruction}`。
  - `writer_context_view(section_id)` → 仅 `{claim_units[], source_registry_slice[], claim_strength, boundary, citation_map, forbidden:[raw_search_result,diagnostic_gap,retry_plan]}`。
  - `repair_context_view(gap)` → 仅诊断槽位 + 下一步搜索任务。
- 数据来自 Phase 1 的 `lookup_evidence_by_requirement` + 内存 claim_units。
- 接入 `agents/analysis_agent.py` / writer 链为**可选输入整形器**：flag 关时维持现有「传全量 state」行为（fail-open）。

**新增 flag**：`CONTEXT_VIEW_BUILDER_ENABLED`（默认 **false**）。

**验证**：flag 开跑 golden case → analysis/writer 产出与基线等价（允许等价不等同）；`telemetry/context_budget.py` 测得的 token 数下降。

**回滚**：flag 关。

---

### Phase 5 — 局部重跑 + 命中规则收口 + freshness（解 G-6，兑现降本提速）｜~3 天

**目标**：按 requirement_id 局部重跑；派生缓存命中条件补版本字段；query-intent freshness。

**改动点**
- **5a 局部重跑驱动**：`flows/report/full_report.py` 增 `--rerun-requirements R1,R2` + `--rerun-run-id <id>`：从 stage_snapshot 复原其余产物，仅对指定 requirement 重跑「搜索→抽取→入库→重建受影响 claim_unit/section」。依赖 Phase 1 索引 + snapshot replay（`load_stage_snapshot`）。
- **5b 派生命中补字段**：stage_snapshot manifest 与 topic_bundle manifest 增 `prompt_version / model / input_hash / producer_version`；claim_unit/section/qa 的「可复用」判定按 §4 表执行（当前它们内嵌在 structured_analysis/writer_report 快照里，补这些字段后才能安全复用）。
- **5c query-intent freshness**：requirement 增 `{freshness_required: bool, max_cache_age_hours: int}`（planning 阶段按查询意图——「今日/最新/实时/行情/融资/财报/政策」——置位）；`evidence_cache.lookup_search`(L502) 与 `topic_bundle.preflight`(L824) 读取并强制；命中超龄则 bypass。
- **5d 复用比护栏**：`cache_report` 增 `cache_reuse_ratio / fresh_search_ratio / mandatory_refresh_slots`；正式行研可配「政策/财报/融资/新闻类 requirement 必须重验」。

**新增 flag**：`FRESHNESS_POLICY_ENABLED`（默认 false）；`LOCAL_RERUN_ENABLED`（默认 false）。

**验证**：对某 golden run 触发单 requirement 重跑 → 确认只有该槽位重搜重抽、其余 cache 命中；freshness 开后「最新」类查询不再命中旧缓存。

**回滚**：flag 关 → 回全量跑。

---

### Phase 6 —（仅当触发）引擎迁移：Postgres + 对象存储 + Redis｜不在当前范围

见 §9 触发条件。前提：Phase 1–5 已把读写收口到 `cache_registry` 门面（§3.2），迁移只换 adapter，不动业务逻辑。

#### 3.2 cache_registry 门面（贯穿 Phase 1–5 逐步收口）
新增 `rag_pipeline/cache/cache_registry.py`：对外暴露 `get_evidence / store_evidence / get_bundle / store_snapshot / lookup_by_requirement / cache_report` 等统一方法，内部委托现有 5 存储。**目的**：①消除「5 存储散落直连」②为 Phase 6 引擎迁移留单一替换点。各 Phase 把新读写走 registry，不强制一次性重构旧调用点。

---

## 4. 缓存命中规则总表（命中=可复用的充要条件）

| 对象 | 存储 | 命中键/条件 | 失效触发 | freshness |
|---|---|---|---|---|
| search result | search_cache | hash(归一query+全部搜索参数) | 参数变 / TTL 过期 | TTL 按源类型；query-intent 覆盖(5c) |
| page/fact (evidence) | evidence_cache | `evidence_identity` 命中 **或** requirement_id 索引(1d) | content_hash 变 / extractor_version·quality_contract_version 变(5b) / TTL | TTL 按源类型 |
| trusted source | trusted_source | `evidence_identity` + level≥B | 同上 + 来源族变 | last_verified_at |
| topic bundle (派生) | topic_bundle | topic_key + schema_version + 未污染 + 未超龄 | schema_version / prompt_version / model 变(5b) / age>max | max_age_days(默认30) |
| claim_unit (派生) | structured_analysis 快照 | requirement_id + usable_fact_card_set_hash + analysis_prompt_version + model + schema_version | 任一变 | — |
| section (派生) | writer_report 快照 | claim_unit_hash + layout_version + writer_prompt_version + citation_style_version | 任一变 | — |
| qa result (派生) | qa_result 快照 | section_hash + source_registry_hash + qa_rule_version | 任一变 | — |

铁律（写进 G4 守卫）：**claim_unit/section/final prose/qa_result 永不作为 fact 来源**；只有 `artifact_class=fact` 的对象可进取证上下文。

---

## 5. 数据契约

**LineageEnvelope**（随每条 evidence 顺流，落每个持久层）
```
run_id, report_id, requirement_id, source_id, fact_id,
search_task_id, chapter_id, hypothesis_id, claim_id, proof_role
```
来源：requirement_id/proof_role 生于 planning（`brain_agent.py:1068` 一带）→ 挂 search_task → 挂 fact_card（`analysis_agent.py:2045`）。脊柱工作 = **在 3 个持久边界不丢它**（evidence_cache 写、trusted_source 写、lineage insert）。

**evidence_identity**：`sha256(canonical_url + normalized_fact[:360] + metric + period + value)`（统一两库）。

**source_level 单一权威**：`evidence_quality.classify_evidence`；其余处委托，不自算。

**版本字段**（用于派生失效）：`schema_version / prompt_version / model / extractor_version / quality_contract_version / producer_version / input_hash`。

---

## 6. freshness 策略（query-intent，解 G-6）

| requirement 类型 | max_cache_age | 命中策略 |
|---|---|---|
| 今日/最新/实时/行情/价格 | ≤24h（甚至强制刷新） | 超龄 bypass，强制重搜 |
| 融资/政策/监管/新闻 | 1–3 天 | 重验 |
| 行业报告/白皮书 | 7–30 天 | 复用 |
| 年报/招股书/标准 | 长期 | 复用，保留版本与发布日期 |
正式行研：关键证据 ≥30% 必做 freshness check；政策/财报/融资/新闻类 requirement 必重验。

---

## 7. 风险 register

| 风险 | 缓解 |
|---|---|
| 缓存旧信息致报告过时 | §6 freshness 策略 + mandatory_refresh_slots |
| 派生污染事实层 | G4 artifact_class 守卫 + §4 铁律 |
| schema 过早固化 | 全程 SQLite+文件，加性迁移；引擎迁移延到 Phase 6 |
| 命中过度致新证据进不来 | cache_reuse_ratio / fresh_search_ratio 护栏(5d) |
| 双编排层（full_report→run_brain_agent）改漏 | 读写收口到 cache_registry(3.2)；两层都走门面 |
| 改动破坏现网 | §0 六条契约；每阶段冒烟回归 |

---

## 8. 验证护栏（每阶段必跑）

- **端到端**：`python run_full_report.py "<golden query>" --route web --output-dir output/_verify_<phase>`
- **冒烟**：`python tools/run_split_pipeline_smoke.py`（split 管线）；`python tools/eval_reports.py`（质量回归）
- **基线 diff**：`tools/cache_baseline.py`（Phase 0 建）对比 `cache_report.json` + 正文。
- **SQLite 抽查**：`sqlite3 output/cache/evidence.sqlite "SELECT requirement_id,count(*) FROM evidence_cache GROUP BY requirement_id"`（确认脊柱落地）。
- **golden 集**：`golden_cases/minimal_cases.json`。
- 通过标准：正文无回归（或可解释的改进）、cache_report 指标合理、各 validation 门违规可解释。

---

## 9. 引擎迁移触发条件（满足任意 3 个再做 Phase 6）

多用户并发生成报告 / 多 worker 并发 / 任务状态需重启恢复 / 小程序绑定 user_id·report_id / 需后台查历史报告 / 需对 fact_card·requirement 复杂筛选 / SQLite 写锁影响速度 / 需权限·审计·备份·迁移。

目标架构：FastAPI → PostgreSQL(业务表+ledger+fact_card元数据) → OSS/COS/MinIO(page/raw/markdown/large json) → Redis(队列/lock/限流/短TTL) → Qdrant(向量)。

---

## 10. 执行顺序速查

Phase 0(护栏) → **Phase 1(requirement_id 脊柱)** → Phase 2(统一身份裁决) → Phase 3(硬校验门) → Phase 4(context view) → Phase 5(局部重跑+freshness) →（条件触发）Phase 6(引擎迁移)。

降本提速主要来自 Phase 1+5；质量保护来自 Phase 3+4；Phase 2 是去重对账的地基。**不要先做命中率优化**——脊柱和校验门没就位时优化命中率会放大错误。
