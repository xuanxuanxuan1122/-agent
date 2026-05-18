# Current RAG Pipeline

## 项目简介

这是一个本地 RAG 检索与问答项目，核心流程包括：

1. 原始文本清洗与切分
2. 向量化与 Qdrant 入库
3. 混合检索、证据筛选与答案生成
4. 评测、审计与存储校验

这次已经把原来较分散的大脚本按功能整理进 `rag_pipeline/` 子包，方便后续维护和继续拆分。

项目现在只会自动加载根目录的 `.env`，模型路径、URL、API Key 等都统一放在这里管理。

## 目录结构

```text
current_rag_pipeline/
├─ rag_pipeline/
│  ├─ agents/        # LangGraph 大脑 Agent、本地 RAG 子 Agent、联网分析子 Agent
│  ├─ config/        # 配置项
│  ├─ ingest/        # 切分、embedding、Qdrant 入库
│  ├─ pipelines/     # 高层流程编排
│  ├─ search/        # 检索、记忆、反思、审查、综合回答
│  └─ tools/         # benchmark、校验、审计工具
├─ start_ingest.ps1              # 切片 + 向量化 + 入库启动脚本
├─ start_store.ps1               # 跳过切片，直接把现有 chunks/embedded 数据写入 Qdrant
└─ start_rag.ps1                 # 常用启动脚本
```
## 主要模块说明

### `rag_pipeline/agents`

- `brain_agent.py`
  - 行业研究多智能体系统的主 Agent / Supervisor，负责问题路由、并行调度本地 RAG 与联网分析两个子智能体，并输出供下游 Analysis Agent 使用的结构化决策包
- `rag_agent.py`
  - 基于 LangGraph `StateGraph` 封装的行研 RAG 子智能体，可作为多智能体系统里的本地知识库检索与证据回答节点
- `web_analysis_agent.py`
  - 基于阿里云 IQS Skills 的联网分析子智能体，用于最新信息、网页读取、政策/新闻/行情检索和事实核验

### `rag_pipeline/config`

- `search_config.py`
  - 搜索、回答、reflection、trace 等默认配置

### `rag_pipeline/ingest`

- `slicing.py`
  - 原始文本清洗、分块、生成 `*.chunks.json`
- `embedding_qdrant.py`
  - 本地 embedding、BGE-M3、Qdrant collection 与 upsert 逻辑

### `rag_pipeline/pipelines`

- `ingest_pipeline.py`
  - 把切分和向量化串成一个完整流程

### `rag_pipeline/search`

- `engine.py`
  - 主搜索入口，负责检索、rerank、evidence、answer
- `memory.py`
  - 会话记忆、上下文改写、OpenAI-compatible 请求帮助函数
- `models.py`
  - 数据模型定义
- `reflection.py`
  - 多跳补检索判断
- `review.py`
  - 回答审查
- `synthesis.py`
  - 基于证据综合回答
- `trace.py`
  - trace 落盘与观测

### `rag_pipeline/tools`

- `benchmark.py`
  - 检索/embedding benchmark
- `validate_qdrant_store.py`
  - 校验 Qdrant 存储状态
- `audit_search_qdrant_defs.py`
  - 用于检查搜索模块重复定义等问题

## 当前推荐使用方式

### 1. 数据入库

```powershell
.\.venv\Scripts\python.exe -m rag_pipeline.pipelines.ingest_pipeline
```

或者直接用专门的启动脚本：

```powershell
.\start_ingest.ps1
```

如果 `rag_chunks_store/` 里已经有现成的 `*.chunks.json` 和 `*.chunks.embedded.json`，可以直接跳过切片：

```powershell
.\start_store.ps1
```

### 2. 执行搜索

```powershell
.\.venv\Scripts\python.exe -m rag_pipeline.search.engine "你的问题"
```

也可以直接启动脚本后按提示输入整行问题：

```powershell
.\start_rag.ps1
```

如果希望显式传入问题，推荐这样：

```powershell
.\start_rag.ps1 --query 可可资本 简介
```

`start_rag.ps1` 默认会进入 LangGraph 大脑 Agent 问答模式：先判断问题该走本地知识库、联网分析还是两者协作；当走双子 Agent 协作时，本地 RAG 和联网分析会在同一个 superstep 并行执行，随后由 Supervisor 输出结构化决策包。需要调试原始检索、证据、trace、模型调用信息时，再显式加 `--json`：

```powershell
.\start_rag.ps1 --json --query 可可资本 简介
```

如果要提速，推荐启动常驻 Agent 服务，让本地 embedding 模型保持加载：

```powershell
.\start_rag.ps1 serve
```

服务启动后，可以打开 `http://127.0.0.1:7860` 在浏览器里提问，或用命令行：

```powershell
.\ask_agent.ps1 --query 可可资本 简介
```

常驻服务的第一次提问可能仍会加载模型；从第二次开始会复用已加载的 Qwen/BGE 模型，响应时间会明显下降。默认回答会同时展示“本次使用的本地证据”。

### LangGraph 多智能体集成

项目现在暴露了一个可编排的 LangGraph 主图：

```python
from rag_pipeline.agents.brain_agent import create_brain_agent_graph

brain_agent = create_brain_agent_graph()
state = brain_agent.invoke({"query": "现在机器人行情怎么样", "route": "auto"})
print(state["answer_text"])
```

当前主图会调度两个子智能体：

```python
from rag_pipeline.agents.brain_agent import create_brain_agent_tool
from rag_pipeline.agents.rag_agent import create_rag_agent_tool
from rag_pipeline.agents.web_analysis_agent import create_web_analysis_tool

brain_tool = create_brain_agent_tool()
rag_tool = create_rag_agent_tool()
web_tool = create_web_analysis_tool()
```

大脑 Agent 可以这样直接运行：

```powershell
.\start_rag.ps1 brain --route auto --query 现在机器人行情怎么样
.\start_rag.ps1 brain --route both --query 结合本地资料和最新网页，分析机器人行业行情
.\start_rag.ps1 brain --route all --query 结合机器人行业和特斯拉TSLA股价分析机会
```

大脑 Agent 默认输出精简报告数据包，字段为 `conclusion`、`financial_data`、`key_data`、`data_gaps` 和 `next_action`，避免把调度轨迹直接混进回答。Supervisor 会先按市场规模、竞争格局、政策监管、技术产业链、资本动态 5 个维度计算覆盖率；不足时最多补充检索 3 轮，并把问题下发给 RAG、IQS 或两者并行。需要调试完整 Supervisor 过程时，可加 `--output-mode supervisor_json` 或 `--include-raw-child-states`。

补充检索闭环可这样控制：

```powershell
.\start_rag.ps1 brain --route both --supervisor-max-loops 3 --supervisor-max-followup-queries 4 --query 机器人行业投资机会
.\start_rag.ps1 brain --disable-followup-loop --query 现在机器人行情怎么样
```

`route=auto` 会把企业发展、个人发展、职业成长、赚钱致富、金融投资、融资并购等问题默认路由到 `both`，让本地 RAG 和联网分析并行返回。

联网分析子智能体可以这样直接运行：

```powershell
.\start_rag.ps1 web --query 机器人行业最新行情
```

行情、财务、融资估值等最新数值统一由 IQS 联网分析子智能体检索。

联网搜索会先做 Query 优化：最多拆成 6 个行研子查询，并按意图调用多个阿里 IQS engine/resource 并行搜索；每个子查询目标召回 50 条，随后进行可信源过滤、去重，并复用当前 `qwen3-rerank` 精排模型保留 10 条高质量来源。具体资源池在 `.env` 的 `IQS_ENGINE_ROUTE_*` 中配置，不再限制为 `LiteAdvanced`。

如果只想测试联网搜索接口，启动独立 HTTP 服务：

```powershell
.\start_rag.ps1 web-search-api
```

测试搜索接口：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:7870/search -ContentType "application/json; charset=utf-8" -Body '{"query":"机器人行业最新行情"}'
```

需要先在 `.env` 中配置：

```env
ALIYUN_IQS_API_KEY=your-iqs-api-key
```

更多状态字段、节点和集成方式见 [`docs/LANGGRAPH_BRAIN_AGENT.md`](docs/LANGGRAPH_BRAIN_AGENT.md)、[`docs/LANGGRAPH_RAG_AGENT.md`](docs/LANGGRAPH_RAG_AGENT.md) 与 [`docs/LANGGRAPH_WEB_ANALYSIS_AGENT.md`](docs/LANGGRAPH_WEB_ANALYSIS_AGENT.md)。

## 模型配置统一位置

统一配置文件：

- [`.env`](/D:/pychram/RAG2/current_rag_pipeline/.env)

推荐规则：

- 所有项目级环境变量统一写在 `.env`
- 不再额外维护其他 env 配置文件
- `start_rag.ps1`、`start_ingest.ps1`、`start_store.ps1`、`ask_agent.ps1` 和 `python -m rag_pipeline...` 都会自动读取 `.env`

## 生成阶段上下文优化

检索结果进入 LLM 之前，会先经过一层上下文工程：

- 相关性高的核心证据优先放在前面，减少重要证据被淹没。
- 长证据会按当前问题抽取关键句，避免把整段噪声塞进 prompt。
- 高度重复的证据会被去重，降低重复信息对答案的干扰。
- 会按 token 预算打包上下文，并在 JSON 输出和 trace 中记录 `context_engineering` 统计。

## 检索与精排默认策略

当前默认策略是先扩大召回，再收敛到少量高质量证据：

- 初召回候选：`RAG_TOP_K=5` × `RAG_CANDIDATE_MULTIPLIER=5`，即每个 query 召回约 25 条候选。
- 精排开关：`RAG_ENABLE_API_RERANK=1`，默认使用已配置的外部 rerank 模型。
- 精排输入上限：`RAG_RERANK_MAX_DOCS=25`。
- 精排输出：`RAG_RERANK_TOP_N=5`。
- 最终进入回答和展示的证据：`RAG_EVIDENCE_TOP_K=5`、`RAG_ANSWER_EVIDENCE_TOP_K=5`；严格筛选不足 5 条时，会从 rerank 后候选中补足到 5 条。

常用调参项都在 `.env`：

```env
RAG_LLM_CONTEXT_MAX_TOKENS=6000
RAG_LLM_CONTEXT_MAX_TOKENS_PER_EVIDENCE=900
RAG_LLM_CONTEXT_DEDUP_THRESHOLD=0.86
```

### 3. 跑 benchmark

```powershell
.\.venv\Scripts\python.exe -m rag_pipeline.tools.benchmark
```

### 4. 校验 Qdrant

```powershell
.\.venv\Scripts\python.exe -m rag_pipeline.tools.validate_qdrant_store
```

## 后续维护建议

1. 新增功能时，优先放到 `rag_pipeline/` 对应子目录。
2. 统一使用 `python -m rag_pipeline...` 方式运行，不再回退到旧脚本名。
3. 如果 `search/engine.py` 继续变大，下一步优先再拆成：
   - `retrieval.py`
   - `ranking.py`
   - `evidence.py`
   - `answering.py`
   - `cli.py`
4. 文档类文件后续可以继续整理到 `docs/`，样例数据整理到 `samples/`。

## 一句话总结

现在这套结构已经整理成“按功能分层 + 模块化启动”的模式，后面再迭代会比之前好维护很多。
