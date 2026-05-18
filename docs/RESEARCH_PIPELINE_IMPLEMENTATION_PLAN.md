# 行研 Agent 研究流水线改造计划

本文档用于把“研究流水线”方案落到当前 `RAG2/current_rag_pipeline` 代码中。目标不是再做一个更会写的 Writer，而是把系统固化成：

> 用户问题 -> 研究画像 -> 可验证假设 -> 问题型章节 -> 证据目标 -> 搜索任务 -> RAG/IQS 证据 Lane -> 证据池 -> 证据审计 -> 覆盖率判断 -> 定向补证 -> 分析判断 -> 章节组织 -> 决策/风险 -> 最终写作 -> QA -> 必要时补证或重写

核心原则：

1. 问题不能直接进入 Writer。
2. Writer 不能新增事实。
3. 章节不能来自固定模板。
4. 搜索必须服务具体证据目标。
5. 证据必须绑定章节、假设、来源等级和证明角色。
6. 证据覆盖不足时先补证或降级表达，不能硬写强结论。
7. 最终正文不能泄露内部过程语言，例如“章节判断”“证据深读”“机制与边界”“进入综合决策章的变量”“evidence_refs”“claim_status”等。

---

## 1. 当前代码总体判断

当前项目已经不是“一个 Writer 写到底”的结构，已经具备研究流水线雏形。主要模块已经存在：

| 流水线阶段 | 当前代码位置 | 当前状态 |
| --- | --- | --- |
| 入口与路由 | `rag_pipeline/agents/brain_agent.py` | 已有 `route_query`、`_route_agents`，可区分 local/web/both/all |
| 问题框定 | `problem_framing_agent.py` | 已能生成 core_question、decision_context、hypotheses |
| 研究规划 | `research_planner.py` | 已强调假设、证据目标、search_task 单目标、禁止固定五维 |
| 动态搜索结构 | `dynamic_search_schema.py` | 已有 Hypothesis、Chapter、EvidenceGoal、SearchTask 结构 |
| Pre-layout | `planning/pre_layout_agent.py` | 已能把 legacy fixed title 改成问题型章节，并补 evidence mix 和最低证据门槛 |
| IQS 检索 | `web_analysis_agent.py` | 已有 search_task 驱动 query plan、fallback、过滤、重排、可信度评分 |
| 本地 RAG | `rag_agent.py` | 定位是本地知识库证据型回答 |
| 证据清洗 | `evidence_merger.py` | 已有 source level、污染过滤、metric 抽取、evidence role |
| 证据绑定 | `evidence/evidence_binder.py` | 已有 source registry、chapter packages、coverage matrix、conflicts |
| 覆盖率监督 | `brain_agent.py` | 已有 coverage units、coverage score、follow-up query、补证循环 |
| 分析与判断 | `analysis_agent.py`、`argument/claim_builder_agent.py` | 已有 evidence -> claim、public claim 过滤 |
| 微布局 | `layout/micro_layout_agent.py` | 已有 chapter block 和 table plan 概念 |
| 表格 | `table/table_agent.py`、`table/table_validator.py` | 已有证据支撑表格、表格校验 |
| 章节组织 | `argument/chapter_argument_agent.py` | 已能按公开 argument/table 聚合章节 |
| 决策与风险 | `synthesis/decision_synthesis_agent.py`、`synthesis/risk_agent.py` | 已有综合判断、风险包 |
| 最终写作 | `writer/final_writer_agent.py`、`writer/markdown_renderer.py` | 已强调只渲染结构化包 |
| Public Sanitizer | `writer/public_report_sanitizer.py` | 已处理内部标签和不适合公开的过程语言 |
| QA/Rewrite | `qa/qa_agent.py`、`qa/rewrite_agent.py` | 已有内部标签、深度、证据、表格、机制、反证等检查 |
| 全链路 Writer | `writer_agent_clean.py` | 已串联 Binder、Synthesizer、Micro Layout、Table、Claim、Decision、Risk、QA、Rewrite |

结论：

当前架构方向是正确的，但还存在几个导致“看起来仍像固定模板”“搜索证据少”“没有真正闭环”的关键问题：

1. **研究请求画像还不是一等状态对象**：现在 route 和 query_analysis 有相关判断，但缺少统一的 `research_request_profile` 合同。
2. **Problem Framing 的证明强度不够动态**：投资/进入/公司尽调等高风险问题应自动升级为 strong proof 和强制 counter evidence。
3. **Planner 的兜底路径偏弱**：当 LLM Planner 未启用或失败时，`research_planner` 可能只返回 seed plan，章节和 search_tasks 主要依赖后续补全，容易变浅。
4. **部分输出分支可能绕过覆盖率闭环**：`brain_agent.merge_outputs_node` 的 `agent_text` 分支默认走自我精炼和 writer pipeline，容易弱化统一的 `run_supervisor_evidence_loop`。
5. **Evidence Pool 合同不够强**：RAG、IQS、follow-up 的结果虽然能汇总，但不是所有字段都被硬性要求保留到 Binder。
6. **覆盖率结果没有完全反向驱动 QA 补证**：QA 能识别 required_followups，但它们还没有稳定回流到 Brain 的补证循环。
7. **内部分析 Lens 仍有泄露风险**：`Claim Builder` 内部会生成“机制拆解”“反证边界”“决策含义”等分析块，虽然渲染层已有清洗，但应在包结构层明确标记为 internal lens，而不是可渲染标题。
8. **风险包和决策包还可以更结构化**：尤其 trigger、impact、watch_metric、decision boundary 应可被 Markdown Renderer 明确渲染，而不是混在描述里。
9. **证据深度和 20000 字级报告目标缺少统一调度策略**：需要把目标篇幅转化为章节数量、每章 argument units、证据目标数量、搜索轮次和最低可用来源数。

---

## 2. 目标架构

### 2.1 总链路

目标链路固定为：

1. `Brain / Router`
   - 生成研究请求画像。
   - 判断 route：local、web、both、all。
   - 不写报告。

2. `Problem Framing`
   - 把用户问题拆成可验证假设。
   - 输出 core question、decision context、hypotheses、proof standards、falsification triggers。
   - 不搜索、不写正文。

3. `Research Planner`
   - 基于画像和假设生成问题型章节、证据目标、搜索任务。
   - 不套固定模板。
   - 不写正文。

4. `Pre-layout`
   - 把研究计划规范成报告蓝图。
   - 补充每章 evidence mix、min sources、counter requirement、metric requirement。
   - 不写正文。

5. `RAG + IQS Evidence Lanes`
   - RAG 负责内部资料和本地知识库。
   - IQS 按证据类型 Lane 检索官方数据、公司披露、市场研究、新闻事件、技术产品、客户案例。
   - 不下最终结论。

6. `Evidence Pool`
   - 所有候选证据统一进入证据池。
   - 保留 task_id、chapter_id、hypothesis_id、evidence_goal_id、proof_role、lane、query、source、raw fact、metric。

7. `Evidence Merger / Binder`
   - 清洗、去重、评级、绑定章节和假设。
   - 生成 source_registry、chapter_evidence_packages、metric_normalization_table、coverage_matrix、conflicts。
   - 不进行主观扩写。

8. `Coverage Supervisor`
   - 判断每章和每个假设是否够证据。
   - 不够时生成定向补证任务。
   - 达标或连续无增益后才进入分析。

9. `Analysis / Claim Builder`
   - 把证据转换成判断单元。
   - 内部可分析核心判断、机制、反证边界、决策含义，但这些标签不能作为正文标题。
   - 不编造新事实。

10. `Micro Layout / Table / Chapter Argument`
    - 规划每章论证顺序。
    - 只生成服务判断的表。
    - 组织章节正文包。

11. `Decision / Risk`
    - 汇总机会判断、行动建议、加码条件、降级条件、风险触发器。
    - 绑定假设和证据边界。

12. `Final Writer`
    - 只渲染结构化包。
    - 不新增事实、不新增结论。

13. `Public Sanitizer / QA / Rewrite`
    - 检查内部标签泄露、证据支撑、结构连贯、表达质量、表格质量、报告深度。
    - 表达问题走 Rewrite。
    - 证据问题回到补证或降级表达。

---

## 3. 关键数据合同

### 3.1 ResearchRequestProfile

建议新增或固化为状态字段：

```python
research_request_profile = {
    "research_object": "...",
    "object_type": "industry | company | product | policy | technology | investment_target | unknown",
    "decision_scenario": "investment_screening | market_entry | product_planning | due_diligence | policy_impact | market_scan | strategy",
    "time_scope": "latest | last_3_months | last_12_months | annual | historical | unspecified",
    "geo_scope": "china | global | overseas | province_city | unspecified",
    "output_depth": "quick_scan | standard_industry_report | deep_industry_report | investment_due_diligence",
    "source_mode": "local | web | both | all",
    "risk_level": "normal | high_accuracy | investment | legal_policy | financial",
    "freshness_required": True,
    "evidence_strictness": "balanced | strict | investment_grade",
}
```

落点：

- `rag_pipeline/agents/brain_agent.py`
  - 在 `build_query_analysis` 或单独 `build_research_request_profile` 中生成。
  - 写入 `BrainAgentState["research_request_profile"]`。
- `rag_pipeline/agents/problem_framing_agent.py`
  - 接收该 profile，动态决定 proof_standard 和 counter requirement。
- `rag_pipeline/agents/research_planner.py`
  - 接收该 profile，决定章节密度、证据目标数量、搜索深度。

验收：

- “最新、近期、行情、政策、公告、财报、融资、价格、股价”自动 `freshness_required=True` 且 route 至少 web。
- “行业研究、投资机会、市场进入、公司分析”默认 both。
- 投资/进入/尽调场景默认 high_accuracy 或 investment_grade。

### 3.2 Hypothesis

当前 `dynamic_search_schema.Hypothesis` 已接近目标。需要强制字段：

```python
hypothesis = {
    "hypothesis_id": "H1",
    "statement": "...",
    "must_prove": ["..."],
    "must_disprove": ["..."],
    "proof_standard": "strong | medium | weak",
    "counter_evidence_required": True,
    "required_source_levels": ["A", "B"],
    "required_evidence_types": ["metric", "case", "filing", "policy", "counter"],
    "metric_definitions": [
        {"metric_name": "...", "scope": "...", "period": "...", "unit": "..."}
    ],
    "falsification_triggers": ["..."],
    "decision_relevance": "...",
}
```

调整重点：

- `problem_framing_agent._hypothesis` 当前默认 `proof_standard="medium"`、`counter_evidence_required=False`，需要按场景动态升级。
- 投资、市场进入、公司尽调、政策影响类核心假设必须：
  - `proof_standard="strong"`
  - `counter_evidence_required=True`
  - `required_source_levels=["A","B"]`
  - 至少包含一个 falsification trigger。

### 3.3 ChapterBlueprint

章节必须是问题型对象，而不是固定目录名：

```python
chapter = {
    "chapter_id": "C1",
    "title": "这个需求是真实采购，还是概念热度？",
    "core_question": "...",
    "related_hypotheses": ["H1", "H2"],
    "evidence_goals": ["EG1", "EG2"],
    "required_evidence_mix": ["official_data", "filing_company", "market_research", "customer_case", "counter_evidence"],
    "min_total_sources": 12,
    "min_ab_sources": 4,
    "min_counter_sources": 1,
    "metric_requirements": ["scope", "period", "unit"],
    "table_policy": "only_if_evidence_supported",
}
```

调整重点：

- `planning/pre_layout_agent.py` 已有 legacy title rewrite，但应进一步避免把 fallback 写成固定“当前规模、竞争强度、政策传导”等标准句。
- 章节标题应优先来自：
  1. 用户问题画像；
  2. 核心假设；
  3. 证据目标；
  4. 决策场景。
- legacy fixed title 只作为纠偏输入，不应成为常规输出。

### 3.4 EvidenceGoal

每个章节必须有多个可验证证据目标：

```python
evidence_goal = {
    "goal_id": "EG1",
    "chapter_id": "C1",
    "hypothesis_id": "H1",
    "goal_question": "是否存在近两年的真实采购、订单、中标或交付证据？",
    "evidence_type": "case | metric | filing | policy | counter | source_check",
    "proof_role": "support | metric | case | counter | source_check",
    "required_source_level": "A/B",
    "minimum_items": 2,
}
```

验收：

- 每个核心章节至少 3 个 evidence goals。
- 投资/进入相关章节必须有 counter goal。
- 市场规模/价格/渗透率章节必须有 metric goal 和 source_check goal。

### 3.5 SearchTask

SearchTask 是检索层最重要的执行单元：

```python
search_task = {
    "task_id": "T1",
    "chapter_id": "C1",
    "hypothesis_id": "H1",
    "evidence_goal_id": "EG1",
    "query": "...",
    "proof_role": "support | metric | case | counter | source_check",
    "lane_targets": ["customer_case", "news_event"],
    "must_have_terms": ["液冷", "AI服务器"],
    "forbidden_terms": ["五粮液", "手机散热"],
    "min_source_level": "B",
    "freshness": "latest | recent | stable | historical",
    "source_priority": ["official", "filing", "association", "research", "news"],
}
```

验收：

- 一个 SearchTask 只能服务一个 evidence goal。
- 必须保留 `task_id/chapter_id/hypothesis_id/evidence_goal_id/proof_role` 到 Evidence Pool。
- IQS query variants 必须继承原始 task metadata。

### 3.6 EvidencePoolItem

所有 RAG、IQS、follow-up 结果统一进入 Evidence Pool：

```python
evidence_pool_item = {
    "round": 0,
    "agent": "rag | official_data | filing_company | market_research | news_event | technology_product | customer_case",
    "query": "...",
    "task_id": "T1",
    "chapter_id": "C1",
    "hypothesis_id": "H1",
    "evidence_goal_id": "EG1",
    "proof_role": "case",
    "answer": "...",
    "fact": "...",
    "raw_data_points": [],
    "key_sources": [],
    "status": "success | partial | failed",
    "confidence": 0.0,
    "limitations": [],
}
```

调整重点：

- 在 `brain_agent` 汇总 RAG/IQS 输出时做统一 schema normalize。
- 不符合最低字段要求的结果可以保留 trace，但不能进入 Binder 核心池。

### 3.7 EvidenceItem

Binder 使用的规范证据：

```python
evidence_item = {
    "evidence_id": "E1",
    "source_ref": "S1",
    "source_level": "A | B | C | D",
    "allowed_use": "core | supporting | directional | clue | appendix | rejected",
    "chapter_id": "C1",
    "hypothesis_id": "H1",
    "evidence_goal_id": "EG1",
    "proof_role": "metric | case | counter | source_check",
    "fact": "...",
    "metrics": [
        {"name": "...", "value": "...", "unit": "...", "period": "...", "scope": "...", "method": "..."}
    ],
    "limitations": [],
    "conflicts": [],
}
```

验收：

- 无 source_ref 的证据不能进入核心判断。
- D 级来源不能支撑核心结论。
- C 级来源只能 directional 或 supporting，不能单独支撑投资结论。
- 缺 scope/period/unit 的指标不能作为强结论依据。

---

## 4. 分阶段实施计划

## Phase 0：合同与观测先行

目标：

先把流水线每一步的输入输出固定下来，避免后续继续出现“某一步生成了，但下一步没有吃到”的问题。

涉及文件：

- `rag_pipeline/contracts/package_contracts.py`
- `rag_pipeline/agents/brain_agent.py`
- `rag_pipeline/agents/dynamic_search_schema.py`
- `rag_pipeline/agents/writer_agent_clean.py`

任务：

1. 新增或固化 `ResearchRequestProfile` 合同。
2. 新增 Evidence Pool 最低字段校验函数。
3. 给 `BrainAgentState` 增加明确字段：
   - `research_request_profile`
   - `coverage_history`
   - `followup_history`
   - `qa_followup_tasks`
   - `final_publication_status`
4. 输出 debug snapshot 时按阶段保存：
   - `01_request_profile.json`
   - `02_problem_framing.json`
   - `03_research_plan.json`
   - `04_report_blueprint.json`
   - `05_search_task_schedule.json`
   - `06_evidence_pool_round_*.json`
   - `07_evidence_binder.json`
   - `08_coverage_report.json`
   - `09_argument_units.json`
   - `10_writer_package.json`
   - `11_qa_result.json`

验收标准：

- 任意一份报告可以回看完整链路。
- 可以回答：
  - 哪个章节来自哪个假设？
  - 哪条证据来自哪个搜索任务？
  - 哪个强结论由哪些 A/B 来源支撑？
  - 哪些缺口触发过补证？

---

## Phase 1：研究请求画像与路由强化

目标：

用户问题进入系统后，先生成任务定义，而不是直接搜索或写作。

涉及文件：

- `rag_pipeline/agents/brain_agent.py`
- `rag_pipeline/agents/problem_framing_agent.py`

任务：

1. 在 `brain_agent.py` 增加 `build_research_request_profile(query, context)`。
2. Profile 识别字段：
   - 研究对象；
   - 研究类型；
   - 决策场景；
   - 时间要求；
   - 地域范围；
   - 输出深度；
   - 信息来源；
   - 风险等级；
   - 是否需要最新事实。
3. route 规则调整：
   - 最新、近期、当前、政策、监管、财报、公告、融资、价格、股价、行情 -> web。
   - 本地资料、知识库、内部文档、已有材料、访谈 -> local。
   - 行业研究、投资机会、市场进入、公司发展、金融投资 -> both。
   - 深度尽调或用户明确全量 -> all。
4. Profile 写入 state，并传递给 planner、writer constraints、QA。

验收标准：

- “新能源汽车新型材料行情怎么样”识别为：
  - object_type=industry/product/materials
  - decision_scenario=market_scan 或 industry_research
  - freshness_required=True
  - route=web 或 both
  - output_depth 至少 standard，用户要求深度时为 deep。
- “某公司值不值得投”识别为 investment_due_diligence，proof strictness 升级。
- Brain 不在该阶段生成正文。

---

## Phase 2：Problem Framing 改成动态假设引擎

目标：

把模糊问题变成可证明、可推翻、可检索的假设集。

涉及文件：

- `rag_pipeline/agents/problem_framing_agent.py`
- `rag_pipeline/agents/research_planner.py`
- `rag_pipeline/agents/dynamic_search_schema.py`

当前问题：

- `_hypothesis` 默认 proof_standard 偏 medium。
- `counter_evidence_required` 默认 False。
- 部分新能源汽车材料逻辑较具体，但其他行业主要靠泛化规则。

任务：

1. 增加场景型 hypothesis profile：
   - 行业机会；
   - 投资筛选；
   - 市场进入；
   - 公司尽调；
   - 政策影响；
   - 技术路线；
   - 产品规划；
   - 价格/行情跟踪。
2. 每类 profile 自动生成四类基础假设：
   - 需求假设；
   - 供给/能力假设；
   - 商业化/利润假设；
   - 风险/反证假设。
3. 投资、进入、尽调、政策影响类强制：
   - strong proof；
   - counter evidence required；
   - A/B source requirement；
   - falsification triggers。
4. 每个 hypothesis 输出 `must_prove` 和 `must_disprove`。
5. 每个 hypothesis 关联 `required_evidence_types`。

验收标准：

- 用户问“有没有机会”“值不值得投”“能不能进入”时，不再生成泛泛背景章节，而是生成可验证假设。
- 每个核心假设都能被后续 evidence_goal 和 search_task 追踪。
- 没有反证的投资判断不能进入强结论。

---

## Phase 3：Research Planner 生成可执行计划，而不是目录

目标：

Planner 输出核心问题、假设、问题型章节、证据目标、搜索任务。即使 LLM Planner 关闭，也要有高质量 deterministic plan。

涉及文件：

- `rag_pipeline/agents/research_planner.py`
- `rag_pipeline/agents/dynamic_search_schema.py`
- `rag_pipeline/agents/planning/pre_layout_agent.py`

当前问题：

- `RESEARCH_PLANNER_SYSTEM` 的设计正确。
- 但 LLM Planner 未启用或失败时，`_dynamic_seed_plan` 可能章节、evidence_goals、search_tasks 偏少。
- 这会导致后续搜索任务不够深，最终报告证据少。

任务：

1. 增强 deterministic planner：
   - 根据 `research_request_profile` 和 hypotheses 生成 5-8 个问题型章节。
   - 每章 3-5 个 evidence_goals。
   - 每个 evidence_goal 生成 1-3 个 search_tasks。
2. Planner 输出禁止以下固定章节标题作为正常结果：
   - 市场规模；
   - 竞争格局；
   - 政策环境；
   - 技术路线；
   - 风险分析；
   - 关键事实速览；
   - 证据深读；
   - 章节判断；
   - 可引用事实；
   - 机制与边界；
   - 进入综合决策章的变量。
3. 对每个 search_task 强制校验：
   - 只能服务一个 evidence_goal；
   - 必须有 `must_have_terms`；
   - 必须有 `forbidden_terms`；
   - 必须有 `proof_role`；
   - 必须有 `lane_targets`。
4. 在 `dynamic_search_schema.normalize_research_plan` 中增加缺失任务补全：
   - 支撑任务；
   - metric 任务；
   - case 任务；
   - source_check 任务；
   - counter 任务。
5. 输出 plan quality report：
   - chapter_count；
   - evidence_goal_count；
   - search_task_count；
   - counter_task_count；
   - fixed_template_title_count；
   - unbound_task_count。

验收标准：

- 深度报告默认至少：
  - 5 个核心章节；
  - 20 个 evidence_goals；
  - 30 个 search_tasks；
  - 每个核心假设至少 1 个 counter task。
- Planner 失败时不退回固定模板。
- 章节由问题和假设推导，而不是资料分类。

---

## Phase 4：Pre-layout 强化为报告蓝图审计器

目标：

Pre-layout 不写正文，只把计划变成可执行蓝图，同时拦截固定模板。

涉及文件：

- `rag_pipeline/agents/planning/pre_layout_agent.py`

任务：

1. 保留当前 `LEGACY_FIVE_TITLES` 纠偏逻辑，但改为：
   - legacy title 只触发 rewrite；
   - rewrite 必须引用 hypothesis 或 core_question；
   - 不再输出固定改写句。
2. 每章生成：
   - `chapter_question`；
   - `required_evidence_mix`；
   - `min_candidate_sources`；
   - `min_core_sources`；
   - `min_ab_sources`；
   - `min_counter_sources`；
   - `metric_completeness_required`；
   - `table_policy`。
3. 针对 20000 字深度报告，Pre-layout 增加体量规划：
   - 每章目标字数；
   - 每章最少 argument units；
   - 全文表格上限；
   - 附录来源和口径表策略。
4. 输出 layout validation warnings：
   - 固定标题；
   - 没有核心问题；
   - 没有 counter evidence goal；
   - 证据组合过单一；
   - 表格计划过多。

验收标准：

- 报告蓝图中不出现固定模板章节标题。
- 每个章节都有“为什么要写这一章”的研究问题。
- 总结章不再承载“每章分析模板”，而是承载综合判断、决策含义、风险触发器、后续验证。

---

## Phase 5：IQS 搜索深度优化

目标：

联网搜索不是泛搜，而是按 evidence_goal 和 proof_role 做 query plan、检索、筛选、重排、抽取。

涉及文件：

- `rag_pipeline/agents/web_analysis_agent.py`
- `rag_pipeline/agents/brain_agent.py`
- `.agents/skills/alibabacloud-iqs-search` 相关调用逻辑

当前基础：

- `web_analysis_agent.build_llm_query_plan` 已能在 search_task 存在时围绕任务生成 query。
- `task_acceptance_filter` 已检查 must_have、forbidden、source、goal relevance。
- 已有 credibility_score、lexical_relevance_score、rerank。

任务：

1. 强制 full report IQS 只走 search_task 驱动，不走泛化 related_questions 主路径。
2. 每个 SearchTask 生成 3-7 个 query variants：
   - support：事实支撑；
   - metric：指标、口径、单位、期间；
   - case：订单、采购、中标、客户、交付；
   - counter：延期、取消、价格战、需求不及预期；
   - source_check：官方、协会、年报、公告、白皮书、券商。
3. Query variants 必须继承：
   - task_id；
   - chapter_id；
   - hypothesis_id；
   - evidence_goal_id；
   - proof_role；
   - lane_targets。
4. 增加 lane-specific source priority：
   - official_data：政府、统计局、工信部、发改委、协会、标准；
   - filing_company：交易所、公告、年报、招股书、公司财报；
   - market_research：券商、咨询、协会白皮书、数据库；
   - news_event：新闻、处罚、中标、招投标、事故、延期；
   - technology_product：标准、专利、论文、产品手册；
   - customer_case：采购公告、中标公告、客户案例、交付案例。
5. 对深度报告增加结果预算：
   - 每个 query variant 取 top N；
   - 每个 evidence_goal 至少保留候选 5-10 条；
   - 每章进入 Binder 前候选 20-40 条；
   - 去重后再进入 source scoring。
6. 增强内容抽取：
   - 标题、摘要、正文片段；
   - 数字、单位、期间、范围；
   - 公司、产品、项目、客户；
   - 反证事件；
   - 原始 URL 和发布日期。

验收标准：

- 对“新能源汽车新型材料行情怎么样”这类当前行情问题，IQS 不能只有少量泛泛新闻。
- 每个章节都能看到 search_task -> query variant -> result -> evidence item 链路。
- 搜索结果不再大量丢失，因为 must_have 过严或 query anchor 过窄。
- 低质转载、自媒体、无来源内容只作为 clue 或剔除。

---

## Phase 6：RAG 输出并入 Evidence Pool

目标：

RAG 负责内部资料，不直接写报告。RAG 结果必须和 IQS 一样进入 Evidence Pool。

涉及文件：

- `rag_pipeline/agents/rag_agent.py`
- `rag_pipeline/agents/brain_agent.py`
- `rag_pipeline/agents/evidence_merger.py`

任务：

1. RAG 输出统一为：
   - answer；
   - evidence snippets；
   - source file；
   - chunk id；
   - confidence；
   - gaps；
   - conflicts；
   - related task metadata。
2. 如果 RAG query 来自 search_task，则 RAG result 也要绑定：
   - task_id；
   - chapter_id；
   - hypothesis_id；
   - evidence_goal_id；
   - proof_role。
3. 如果 RAG 结果没有来源文件或 chunk，不进入 core evidence。
4. RAG 与 IQS 冲突时不自动合并，交给 Binder 的 conflicts 处理。

验收标准：

- 本地资料可以成为背景和内部证据，但不会绕过 Evidence Binder。
- RAG 不再直接影响 Writer 的事实生成。

---

## Phase 7：Evidence Merger / Binder 强化为事实审计层

目标：

把候选结果变成可用证据，并严格决定哪些证据能支撑核心判断。

涉及文件：

- `rag_pipeline/agents/evidence_merger.py`
- `rag_pipeline/agents/evidence/evidence_binder.py`

当前基础：

- Source scoring 已有 A/B/C/D。
- Binder 已有 source_registry、chapter packages、coverage matrix、metric table、conflicts。

任务：

1. Evidence Merger 强制抽取：
   - fact；
   - metrics；
   - source type；
   - source level；
   - publication time；
   - domain；
   - evidence role；
   - contamination flags。
2. 扩充污染词规则：
   - 同音/歧义行业；
   - unrelated consumer products；
   - forum/self-media/repost；
   - pure SEO pages；
   - missing source pages。
3. Metric normalization 强化：
   - 年份不能误作指标值；
   - 百分比、金额、数量、出货量分类；
   - 单公司订单不能当行业规模；
   - 缺 unit/scope/period 降级。
4. Binder 阈值拆分：
   - `min_candidate_sources`：候选来源数量；
   - `min_core_sources`：可进核心判断来源数量；
   - `min_ab_sources`：A/B 来源数量；
   - `min_counter_sources`：反证数量；
   - `min_complete_metrics`：完整指标数量。
5. Binder 输出：
   - source_registry；
   - chapter_evidence_packages；
   - hypothesis_coverage_matrix；
   - metric_normalization_table；
   - conflicts；
   - missing_proof_standards；
   - followup_recommendations。

验收标准：

- 没有 source_ref 的事实不能进正文核心判断。
- C/D 来源不能单独支撑强结论。
- 每个强判断都可以追溯到 evidence refs。
- 指标口径冲突不会被平均或强行统一，而是保留差异和边界。

---

## Phase 8：Coverage Supervisor 统一闭环

目标：

证据不够时必须先补证或降级，不能直接写。

涉及文件：

- `rag_pipeline/agents/brain_agent.py`
- `rag_pipeline/agents/writer_agent_clean.py`
- `rag_pipeline/agents/qa/qa_agent.py`

当前关键问题：

- `brain_agent.merge_outputs_node` 中默认 `BRAIN_OUTPUT_MODE=agent_text` 时，可能先走 self-refinement 和 writer pipeline，而不是统一的 `run_supervisor_evidence_loop`。
- 这会造成“报告写出来了，但证据闭环不足”的体验。

任务：

1. 将 `run_supervisor_evidence_loop` 提升为所有 full report 输出模式的必经步骤。
2. `agent_text` 分支不应绕过 coverage supervisor。
3. Coverage score 规则：
   - 每章 total；
   - A/B；
   - counter；
   - metric completeness；
   - hypothesis coverage；
   - decision readiness。
4. Follow-up 生成：
   - 缺 A/B -> official/filing/association/research；
   - 缺 metric -> scope/period/unit/统计/口径；
   - 缺 counter -> 延期、取消、价格战、需求不及预期、风险案例；
   - 缺 case -> 客户、订单、中标、采购、交付；
   - 缺 company proof -> 年报、公告、招股书、交易所。
5. Stop conditions：
   - coverage >= 0.8；
   - 关键章节达标；
   - 达到 max loops；
   - 连续两轮无提升；
   - 只剩 minor gap；
   - 外部公开信息确实不足，转边界化表达。
6. Coverage report 写入 debug snapshot。

验收标准：

- 任何进入 Writer 的报告都有 coverage_report。
- 投资/进入/尽调报告没有反证时不能强行输出“建议进入/投资”。
- 补证任务不是重搜大词，而是针对缺口生成。

---

## Phase 9：Analysis / Claim Builder 内部分析与公开表达隔离

目标：

内部可以做深层分析，但正文不出现内部标签。

涉及文件：

- `rag_pipeline/agents/analysis_agent.py`
- `rag_pipeline/agents/argument/claim_builder_agent.py`
- `rag_pipeline/agents/layout/micro_layout_agent.py`
- `rag_pipeline/agents/writer/markdown_renderer.py`
- `rag_pipeline/agents/writer/public_report_sanitizer.py`

任务：

1. Claim unit 分成 internal lens 和 public paragraph：
   - internal_lens.mechanism；
   - internal_lens.counter_boundary；
   - internal_lens.decision_implication；
   - public_claim；
   - public_reasoning；
   - public_evidence_summary；
   - public_boundary_sentence。
2. 禁止把以下词作为正文标题：
   - 核心判断；
   - 机制拆解；
   - 反证边界；
   - 决策含义；
   - 章节分析；
   - 证据深读；
   - 可引用事实；
   - 进入综合决策章的变量；
   - A/B 来源；
   - proof_role；
   - evidence_refs。
3. Micro Layout 只决定叙事顺序：
   - 先讲问题；
   - 再讲事实；
   - 再讲传导；
   - 再讲边界；
   - 再讲决策含义。
4. Markdown Renderer 不渲染 internal lens title，只把其中内容自然融入段落。
5. Sanitizer 增加内部过程词黑名单和替代表达。

验收标准：

- 正文读起来是一篇正常行研报告，不像 Agent 中间过程。
- 内部分析结果仍能提升文章深度，但不作为模板小标题出现。
- 总结章节可以参考各章分析，但用自然报告语言表达。

---

## Phase 10：Table Agent 从“每章表格”改为“必要表格”

目标：

表格只服务判断，不为装饰而生成。

涉及文件：

- `rag_pipeline/agents/table/table_agent.py`
- `rag_pipeline/agents/table/table_validator.py`
- `rag_pipeline/agents/writer/markdown_renderer.py`

任务：

1. 表格准入条件：
   - 至少 2 行有效证据；
   - 每行有 row_claim；
   - 每行有 evidence_refs；
   - 服务当前章节问题；
   - 不含 source/ref/evidence 等内部表头；
   - 一章最多一张正文表。
2. 表格类型保留：
   - 指标口径对照表；
   - 客户案例矩阵；
   - 产业链利润池表；
   - 玩家能力矩阵；
   - 技术成熟度表；
   - 风险触发器表；
   - 进入策略表。
3. 总结章默认不强制表格。
4. 表格不足时转附录或不渲染。

验收标准：

- 最终报告不会每章都硬塞表。
- 表格不出现“来源、引用、口径、evidence_refs”等正文内部字段。
- 表格行数过多时进入附录。

---

## Phase 11：Decision / Risk 形成真正的价值出口

目标：

最终建议不是“建议关注”，而是给出优先级、加码条件、放弃条件、跟踪指标和风险触发器。

涉及文件：

- `rag_pipeline/agents/synthesis/decision_synthesis_agent.py`
- `rag_pipeline/agents/synthesis/risk_agent.py`
- `rag_pipeline/agents/writer/markdown_renderer.py`

任务：

1. Decision package 固化字段：
   - overall_judgment；
   - opportunity_level；
   - priority_segments；
   - recommended_actions；
   - validation_sequence；
   - upside_triggers；
   - downgrade_triggers；
   - abandon_conditions；
   - watch_metrics。
2. Risk item 固化字段：
   - risk_type；
   - related_hypothesis；
   - description；
   - trigger；
   - impact；
   - mitigation；
   - watch_metric。
3. Markdown Renderer 明确渲染：
   - 总体判断；
   - 优先方向；
   - 行动顺序；
   - 加码条件；
   - 降级/放弃条件；
   - 风险触发器。
4. 风险必须绑定假设，不再简单列“政策风险、市场风险、竞争风险”。

验收标准：

- 报告结尾能指导投资/进入/产品/战略动作。
- 每个建议都能追溯到前文证据和判断。
- 风险触发后能说明结论如何调整。

---

## Phase 12：Writer / QA / Rewrite 发布门

目标：

Final Writer 只渲染，不创造事实；QA 不通过不能直接发布。

涉及文件：

- `rag_pipeline/agents/writer_agent_clean.py`
- `rag_pipeline/agents/writer/final_writer_agent.py`
- `rag_pipeline/agents/writer/markdown_renderer.py`
- `rag_pipeline/agents/writer/public_report_sanitizer.py`
- `rag_pipeline/agents/qa/qa_agent.py`
- `rag_pipeline/agents/qa/rewrite_agent.py`

任务：

1. Final Writer 输入只允许结构化包：
   - report_blueprint；
   - chapter_packages；
   - table_packages；
   - decision_package；
   - risk_package；
   - appendix_package；
   - source_registry。
2. Writer 不允许：
   - 新增数据；
   - 新增事实；
   - 新增结论；
   - 把弱证据写成强结论；
   - 把内部标签写入正文。
3. QA 检查：
   - 是否回答用户问题；
   - 是否有核心观点；
   - 是否有机制解释；
   - 是否有反证；
   - 是否有行动建议；
   - 是否有风险触发器；
   - 是否有内部语言泄露；
   - 是否有无来源事实；
   - 是否达到目标深度；
   - 是否存在表格硬塞；
   - 是否每章都像模板。
4. QA 不通过处理：
   - 表达问题 -> Rewrite；
   - 内部标签 -> Sanitizer + Rewrite；
   - 证据弱 -> 降级表达或隐藏章节；
   - 核心证据缺失 -> 生成补证任务；
   - 缺反证 -> news_event/customer_case 补证；
   - 表格不合格 -> 删除或转附录；
   - 多轮补证仍不足 -> 明确边界化表达。
5. QA 产生的 required_followups 回流到 Brain coverage loop。

验收标准：

- QA 结果不仅能打分，还能决定“补证、重写、降级、隐藏”。
- 最终 `report_status=final` 必须满足：
  - package validation passed；
  - QA passed 或仅 minor warnings；
  - 无内部标签泄露；
  - 无 unsupported strong claim。

---

## 5. 针对“证据太少、报告不够深”的专项优化

用户当前明确要求：除证据外正文至少约 20000 字，分析要更深，不是证据罗列。

### 5.1 深度报告参数

建议新增 deep report profile：

```python
deep_report_policy = {
    "target_body_chars": 20000,
    "chapter_count": [6, 9],
    "argument_units_per_chapter": [3, 5],
    "evidence_goals_per_chapter": [3, 6],
    "search_tasks_per_chapter": [5, 10],
    "candidate_sources_per_chapter": [20, 40],
    "core_sources_per_chapter": [6, 12],
    "ab_sources_per_chapter": [3, 6],
    "counter_sources_per_key_chapter": [1, 3],
    "body_tables_total": [2, 4],
}
```

落点：

- `pre_layout_agent.py`：生成章节和每章目标体量。
- `brain_agent.py`：据此决定 search task 和补证预算。
- `writer_agent_clean.py`：据此决定 QA 深度目标。

### 5.2 从“证据罗列”到“论证叙事”

每章内部顺序：

1. 先提出本章要解决的判断问题。
2. 给出当前最重要的事实组合。
3. 解释事实之间如何传导。
4. 说明哪些证据加强判断。
5. 说明哪些反证限制判断。
6. 给出对决策的含义。
7. 自然过渡到下一章。

注意：

- 正文不要写“本章判断如下”“机制拆解如下”“证据深读如下”。
- 可以把这些作为内部分析结构，但最终只输出自然段落。

### 5.3 搜索预算与补证预算

深度报告建议：

| 对象 | 最低建议 |
| --- | --- |
| 每个核心 hypothesis | support task + metric task + counter task |
| 每个核心章节 | 5-10 个 search tasks |
| 每个 evidence goal | 3-7 个 query variants |
| 每章候选来源 | 20-40 条 |
| 每章可用来源 | 8-15 条 |
| 每章 A/B 来源 | 3-6 条 |
| 投资/进入章节反证 | 1-3 条 |
| 最大补证轮次 | 3-5 轮 |

---

## 6. 针对“模板感”的专项优化

### 6.1 禁止进入正文的模板/内部词

需要在 Planner、Claim Builder、Renderer、Sanitizer、QA 多层拦截：

- 章节判断
- 关键事实速览
- 证据深读
- 全球口径
- 可引用事实
- 机制与边界
- 进入综合决策章的变量
- 核心判断
- 机制拆解
- 反证边界
- 决策含义
- 本章分析
- evidence_refs
- source_registry
- claim_status
- proof_role
- render_blocks
- A/B 来源不足
- 需要补证
- 证据不足

公开表达替代原则：

| 内部表达 | 公开表达 |
| --- | --- |
| 证据不足 | 现有公开信息更适合做边界化观察 |
| 需要补证 | 后续仍需跟踪连续披露的数据 |
| A/B 来源不足 | 该判断尚缺少更高权威口径的连续验证 |
| 反证边界 | 这一判断的限制在于... |
| 决策含义 | 对进入/投资/产品规划而言... |

### 6.2 章节生成原则

错误：

- 市场规模与增长
- 竞争格局
- 政策环境
- 技术路线
- 风险分析

正确：

- 需求到底来自真实采购，还是来自概念热度？
- 哪些指标能证明行情已经进入可持续阶段？
- 利润最终会留在哪些材料、工艺或客户绑定环节？
- 哪些公司或供应链节点已经拿到可验证订单？
- 什么反证会让当前机会判断降级？
- 决策上应该先验证哪些变量，再决定是否加码？

---

## 7. 测试与验收矩阵

### 7.1 单元测试

新增或强化：

1. `test_research_request_profile.py`
   - route 和 profile 识别。
2. `test_problem_framing_dynamic_hypotheses.py`
   - 投资/进入/尽调场景 proof standard 升级。
3. `test_research_planner_dynamic_plan.py`
   - 无 LLM 时也能生成章节、evidence_goals、search_tasks。
4. `test_search_task_contract.py`
   - search_task 单 evidence_goal、metadata 不丢。
5. `test_iqs_query_plan_variants.py`
   - proof_role -> query variants。
6. `test_evidence_pool_schema.py`
   - RAG/IQS/follow-up 输出归一化。
7. `test_evidence_binder_thresholds.py`
   - A/B、counter、metric completeness。
8. `test_coverage_supervisor_loop.py`
   - 缺口 -> follow-up -> 再评估。
9. `test_public_label_sanitizer.py`
   - 内部标签不进入正文。
10. `test_table_admission.py`
   - 表格不足时不渲染。
11. `test_qa_followup_feedback.py`
   - QA required_followups 回流。

### 7.2 Golden Case

至少保留 5 个真实问题作为回归：

1. “现在新能源汽车的新型材料在市场的行情怎么样？”
   - 测试行情、最新、材料、行业研究。
2. “帮我做 AI 服务器液冷行业研究，判断有没有投资机会。”
   - 测试行业机会、投资判断、技术/客户/订单/反证。
3. “某公司值不值得投？”
   - 测试公司尽调、财报、公告、风险。
4. “某政策对行业有什么影响？”
   - 测试政策原文、传导机制、反证。
5. “这个赛道能不能进入？”
   - 测试市场进入、竞争、利润、行动建议。

每个 Golden Case 验收：

- 有 request profile。
- 有 problem framing。
- 有动态章节。
- 有 evidence goals。
- 有 search tasks。
- 有 evidence pool。
- 有 coverage report。
- 有补证记录或明确达标原因。
- 有 source registry。
- 有 metric table。
- 有 claim units。
- 有 decision/risk。
- 正文无内部标签。
- 正文不出现固定小节模板。
- 强结论都有证据支撑。

---

## 8. 推荐执行顺序

### Milestone 1：闭环不绕过

优先级最高。

任务：

1. `agent_text` 分支强制进入 `run_supervisor_evidence_loop`。
2. Evidence Pool schema normalize。
3. Coverage report 必须存在后才能进入 Writer。
4. QA required_followups 可以回流到补证。

完成后解决：

- “没有闭环”
- “证据不够也写”
- “Writer 直接吃搜索摘要”

### Milestone 2：动态规划不退化

任务：

1. ResearchRequestProfile。
2. Problem Framing proof standard 动态化。
3. Deterministic Planner 生成章节、evidence_goals、search_tasks。
4. Pre-layout 拦截固定模板。

完成后解决：

- “还是固定模板”
- “不是先生成模板章节然后根据章节找问题”
- “搜索任务泛泛”

### Milestone 3：搜索和证据深度

任务：

1. IQS search_task-only 主路径。
2. proof_role query variants。
3. lane-specific source priority。
4. 每章候选来源和 A/B 来源预算。
5. Metric 抽取和口径审计。

完成后解决：

- “IQS 没找到证据”
- “证据太少”
- “内容分析不够深”

### Milestone 4：正文自然化

任务：

1. Internal lens 与 public paragraph 分离。
2. Renderer 不渲染内部标题。
3. Sanitizer/QA 增加模板泄露规则。
4. Micro Layout 强化先后逻辑和章节过渡。

完成后解决：

- “章节分析这种出现在正文”
- “核心判断、机制拆解、反证边界、决策含义不要出现在正文”
- “看着硬套模板”

### Milestone 5：决策价值出口

任务：

1. Decision package 字段固化。
2. Risk item 绑定 hypothesis。
3. Renderer 输出加码/降级/放弃/跟踪指标。
4. 表格仅在必要时渲染。

完成后解决：

- “总结章节太浅”
- “只是陈述，没有深层分析”
- “建议不够可执行”

---

## 9. 最终完成标准

一份合格的深度行研报告，应满足：

1. 不是固定模板目录。
2. 每章回答一个真实研究问题。
3. 每章都能追溯到假设、证据目标和搜索任务。
4. 核心判断由 A/B 来源支撑。
5. 投资/进入/尽调判断包含反证和降级条件。
6. 指标有 scope、period、unit。
7. 表格有证据支撑，且不是每章硬塞。
8. 章节之间有自然递进：
   - 为什么要研究；
   - 需求是否成立；
   - 行情是否可持续；
   - 利润在哪里；
   - 谁已经验证；
   - 哪些反证会推翻；
   - 决策上怎么做。
9. 正文无内部过程语言。
10. Writer 没有新增事实。
11. QA 不通过不能直接发布。
12. 报告有完整 debug snapshot，可审计每条事实来源。

---

## 10. 当前最建议马上改的 10 个点

1. 在 `brain_agent.merge_outputs_node` 中统一所有 full report 输出路径，先跑 coverage supervisor，再进 writer。
2. 新增 `research_request_profile` 并传给 Problem Framing、Planner、Pre-layout、QA。
3. 把 `problem_framing_agent._hypothesis` 的 proof standard 改成按场景动态计算。
4. 增强 `research_planner._dynamic_seed_plan`，无 LLM 时也生成完整章节、evidence_goals、search_tasks。
5. 在 `dynamic_search_schema` 中硬校验 search_task 单 evidence_goal 和 metadata 完整性。
6. 在 `web_analysis_agent` 中强制 full report 使用 search_task query plan，减少泛搜 related questions。
7. Evidence Pool 统一 schema，确保 RAG/IQS/follow-up 不丢 task/chapter/hypothesis/evidence_goal。
8. Binder 阈值拆成 candidate/core/A-B/counter/metric，而不是混用一个 min_sources。
9. Claim Builder 的“机制拆解、反证边界、决策含义”改为 internal lens，不允许作为正文标题。
10. QA 的 required_followups 回流补证；无法补足时降级表达，不硬写强结论。

