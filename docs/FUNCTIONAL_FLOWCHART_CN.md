# RAG2 功能流程图

本文档根据当前 `current_rag_pipeline` 项目结构整理，覆盖知识库入库、问答/报告生成、多智能体调度、证据治理、写作质检和 Web 报告入口。

## 1. 功能总览

```mermaid
flowchart TD
    U["用户/业务方"] --> E["运行入口"]

    subgraph EN["启动与服务入口"]
        E --> SR["start_rag.ps1"]
        E --> SING["start_ingest.ps1 / start_store.ps1"]
        E --> RF["run_full_report.py"]
        E --> WEB["report_web_app"]
    end

    SR --> MODE{"选择运行模式"}
    MODE -->|ingest / sync| ING["知识库构建"]
    MODE -->|brain / agent| BRAIN["大脑 Agent 问答/研究"]
    MODE -->|serve| API["本地 RAG 服务"]
    MODE -->|web / api| WSA["联网搜索服务"]
    RF --> BRAIN
    WEB --> RF
    SING --> ING

    subgraph STORE["知识库入库"]
        ING --> CLEAN["原始文本清洗"]
        CLEAN --> CHUNK["语义切片 + 父子块"]
        CHUNK --> EMB["Qwen / BGE-M3 向量化"]
        EMB --> QD["Qdrant Collection"]
    end

    subgraph RUN["问答与报告生成"]
        BRAIN --> PLAN["问题分析 + 研究规划"]
        PLAN --> ROUTE["动态路由"]
        ROUTE --> LRAG["本地 RAG 检索"]
        ROUTE --> IQS["联网 IQS 多通道检索"]
        LRAG --> POOL["统一证据池"]
        IQS --> POOL
        POOL --> LOOP["覆盖率评估 + 补充检索"]
        LOOP --> WRITER["报告写作流水线"]
        WRITER --> QA["QA / Rewrite / Review / Reformatter"]
    end

    QD --> LRAG
    API --> LRAG
    WSA --> IQS
    QA --> OUT["Markdown 报告 / JSON 状态 / 证据包 / Trace"]
```

## 2. 知识库入库流程

```mermaid
flowchart TD
    A["启动入库<br/>start_ingest.ps1 或 start_rag.ps1 ingest"] --> B["加载 .env 配置"]
    B --> C["定位输入目录<br/>RAG_INPUT_PATH"]
    C --> D{"是否跳过切片"}

    D -->|否| E["读取原始 .txt 文档"]
    E --> F["清洗文本<br/>去 HTML / OCR 噪声 / 重复段落 / 低价值内容"]
    F --> G["识别结构<br/>标题 / 段落 / 表格 / 列表 / 指标"]
    G --> H["语义切片<br/>生成子 chunk"]
    H --> I["构建父子块关系<br/>parent_chunk_uid / child_chunk_uids"]
    I --> J["写入 *.chunks.json"]

    D -->|是| J
    J --> K{"是否跳过向量化"}
    K -->|是| Z["结束：仅生成切片文件"]
    K -->|否| L["加载本地 Embedding 模型"]
    L --> M["生成 Qwen dense 向量"]
    M --> N["可选生成 BGE-M3 dense / sparse 向量"]
    N --> O["元数据增强与质量过滤"]
    O --> P{"是否写入 Qdrant"}
    P -->|否| Q["写入 *.embedded.json"]
    P -->|是| R["创建/检查 Qdrant collection"]
    R --> S["批量 upsert 向量和 payload"]
    S --> T["写入 embedding 索引与统计"]
    Q --> T
    T --> U["结束：知识库可检索"]
```

对应模块：

| 功能 | 代码位置 |
|---|---|
| 入库编排 | `rag_pipeline/pipelines/ingest_pipeline.py` |
| 文本清洗与切片 | `rag_pipeline/ingest/slicing.py` |
| 向量化与 Qdrant 写入 | `rag_pipeline/ingest/embedding_qdrant.py` |

## 3. 问答/报告主流程

```mermaid
flowchart TD
    A["用户输入问题/报告主题"] --> B["加载 .env + 运行参数"]
    B --> C{"是否命中 Topic Bundle 缓存"}
    C -->|是| C1["复用缓存证据重建报告"]
    C -->|否| D["run_brain_agent"]
    C1 --> W

    subgraph BG["Brain Agent LangGraph"]
        D --> E["decompose_query<br/>整理问题与上下文"]
        E --> F["route<br/>生成 query_analysis / research_plan / search tasks"]
        F --> G{"select_child_agents"}
        G -->|local| H["industry_rag_agent"]
        G -->|web| I["web_analysis_agent"]
        G -->|all/both| J["6 个 IQS lane"]
        H --> K["本地证据结果"]
        I --> L["联网分析结果"]
        J --> M["角色化联网证据"]
        K --> N["merge_outputs"]
        L --> N
        M --> N
        N --> O["Supervisor 覆盖率评估"]
        O --> P{"证据是否足够"}
        P -->|否| Q["生成 follow-up 查询<br/>回到 RAG/IQS 补证"]
        Q --> O
        P -->|是/停止| R["Evidence Merger<br/>统一证据包"]
        R --> S["Analysis Agent<br/>结构化洞察"]
        S --> T["Writer Pipeline"]
        T --> U["format_response"]
    end

    U --> W["写入阶段快照与状态"]
    W --> X{"报告是否可发布"}
    X -->|可发布| Y["输出正式 Markdown"]
    X -->|需复核| Z["输出评分版/诊断版/Review Draft"]
    Y --> AA["可选 Reformatter 生成 Clean Report"]
    Z --> AB["ReviewAgent 规则清理或人工复核"]
    AA --> AC["最终产物"]
    AB --> AC
```

对应模块：

| 功能 | 代码位置 |
|---|---|
| 完整报告入口 | `rag_pipeline/flows/report/full_report.py` |
| 大脑 Agent 主图 | `rag_pipeline/agents/brain_agent.py` |
| 证据合并 | `rag_pipeline/agents/evidence_merger.py` |
| 写作流水线 | `rag_pipeline/agents/writer_agent_clean.py` |
| 重排版与审查 | `rag_pipeline/flows/report/reformatter_agent.py`、`review_pipeline.py` |

## 4. 本地 RAG 检索子流程

```mermaid
flowchart TD
    A["RAG 查询"] --> B["上下文改写 / Query Plan"]
    B --> C["生成检索变体<br/>关键词 / 别名 / 时间约束 / 任务类型"]
    C --> D["连接 Qdrant"]
    D --> E["查询向量化"]
    E --> F["Dense 召回<br/>Qwen dense"]
    E --> G["Sparse 召回<br/>lexical / BGE sparse"]
    F --> H["层级召回<br/>父块 -> 子块"]
    G --> H
    H --> I["候选合并<br/>RRF / 分数融合 / 去重"]
    I --> J["重排序<br/>API rerank 或 local cross-encoder"]
    J --> K["证据选择<br/>core / support / 引用片段"]
    K --> L["覆盖度与主题一致性检查"]
    L --> M{"证据是否足够"}
    M -->|否| N["Reflection 改写查询<br/>多跳补检索"]
    N --> C
    M -->|是/停止| O["基于证据综合回答"]
    O --> P["答案审查<br/>unsupported claim / 覆盖不足 / 冲突"]
    P --> Q["返回 answer / evidence / trace"]
```

对应模块：

| 功能 | 代码位置 |
|---|---|
| 检索入口 | `rag_pipeline/search/engine.py` |
| 上下文构造 | `rag_pipeline/search/context_builder.py` |
| 多跳反思 | `rag_pipeline/search/reflection.py` |
| 答案综合 | `rag_pipeline/search/synthesis.py` |
| 答案审查 | `rag_pipeline/search/review.py` |
| Trace | `rag_pipeline/search/trace.py` |

## 5. 写作生产与质量治理流程

```mermaid
flowchart TD
    A["Evidence Package / Structured Analysis"] --> B["Pre-layout<br/>报告蓝图"]
    B --> C["Evidence Binder<br/>证据绑定章节/假设/来源等级"]
    C --> D["Evidence Synthesizer<br/>证据图与冲突"]
    D --> E["Analytics Agents<br/>市场/竞品/技术/监管/投资洞察"]
    E --> F["Micro Layout<br/>章节块与表格计划"]
    F --> G["Table Agent<br/>证据支撑表格"]
    G --> H["Claim Builder<br/>公开论证单元"]
    H --> I["Chapter Argument<br/>章节包"]
    I --> J["Decision Synthesis<br/>结论与建议"]
    J --> K["Risk Agent<br/>风险与触发器"]
    K --> L["Final Writer<br/>只渲染结构化包"]
    L --> M["Public Sanitizer<br/>清理内部过程语言"]
    M --> N["QA Agent<br/>证据/结构/引用/表格/深度检查"]
    N --> O{"是否需要改写"}
    O -->|是| P["Rewrite Agent"]
    P --> N
    O -->|否| Q{"交付门是否通过"}
    Q -->|通过| R["final_clean / publishable_clean"]
    Q -->|未通过| S["formal_scored / diagnostic_only"]
    R --> T["报告产物"]
    S --> T
```

关键原则：

- Writer 不直接新增事实，只渲染上游结构化包。
- 证据覆盖不足时，系统优先补证、降级表达或输出诊断/评分版。
- QA 和 Reformatter 主要处理公开表达、引用规范、格式和发布可用性。

## 6. Web 报告应用流程

```mermaid
flowchart TD
    A["浏览器打开 Report Web App"] --> B["填写主标题 / 研究方向 / LLM profile"]
    B --> C["POST /api/reports"]
    C --> D["创建 ReportJob"]
    D --> E["后台线程执行 run_full_report.py"]
    E --> F["实时写入 run.log"]
    E --> G["写入 output/web_reports/{job_id}"]
    G --> H["refresh_artifacts 扫描产物"]
    H --> I{"任务状态"}
    I -->|最终报告可用| J["completed"]
    I -->|需复核草稿/诊断产物| K["needs_review"]
    I -->|失败| L["failed"]
    J --> M["前端下载/预览 final / writer / package / state / log"]
    K --> M
    L --> N["查看日志定位失败原因"]
```

对应模块：

| 功能 | 代码位置 |
|---|---|
| Web API | `report_web_app/main.py` |
| 前端页面 | `report_web_app/static/index.html`、`app.js`、`styles.css` |

## 7. 主要产物

```mermaid
flowchart LR
    A["一次运行"] --> B["*.state.json<br/>全局状态"]
    A --> C["*.writer_package.json<br/>写作包/证据包/质量报告"]
    A --> D["*_report.md<br/>正式报告"]
    A --> E["*_score.md<br/>评分与缺陷清单"]
    A --> F["*_review_draft.md<br/>待复核草稿"]
    A --> G["*_clean.md<br/>可选 Clean Report"]
    A --> H["stage_snapshots/<run_id>/<stage><br/>阶段快照"]
    A --> I["traces / logs<br/>检索与运行轨迹"]
```

## 8. 一句话版本

这个项目的核心功能链路是：

> 原始资料入库到 Qdrant 知识库，用户问题进入 Brain Agent 后被拆成研究计划和搜索任务，本地 RAG 与联网 IQS 并行补齐证据，Evidence Merger 和 Supervisor 做覆盖率闭环，Writer Pipeline 只基于结构化证据生成报告，最后由 QA、Review 和 Reformatter 输出可发布或待复核的 Markdown 产物。
