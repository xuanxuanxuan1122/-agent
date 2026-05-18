# LangGraph 本地 RAG 子智能体

本模块把已有的行业研究 RAG 检索、重排、证据选择和回答生成流程封装成 LangGraph 子智能体，供 `brain_agent` 调度。

## Agent 模块

- `rag_pipeline.agents.rag_agent`
- Agent name: `industry_rag_agent`
- Main graph factory: `create_rag_agent_graph()`
- Programmatic runner: `run_rag_agent(query, session_id="", args_overrides={...})`
- Supervisor/tool entrypoint: `create_rag_agent_tool()`

## 图结构

```mermaid
flowchart LR
    START([START]) --> prepare_query
    prepare_query --> rag_core
    rag_core --> format_response
    format_response --> END([END])
```

### 节点

- `prepare_query`：从 `state["query"]` 或最新 user/human 消息中提取问题。
- `rag_core`：调用已有检索、重排、证据选择、大模型综合、回答审阅、反思、trace 和记忆流程。
- `format_response`：追加 assistant 消息，并返回适合 Supervisor 消费的状态。

## 状态契约

输入字段：

- `query`：用户问题。
- `messages`：可选的消息列表。若 `query` 为空，会读取最新 user/human 消息。
- `session_id`：可选的对话记忆键。
- `args_overrides`：可选的检索引擎参数覆盖，字段名沿用 `rag_pipeline.search.engine` 的 `argparse.Namespace`。

输出字段：

- `answer_text`：带证据约束的最终回答。
- `raw_output`：原始 RAG 引擎输出。
- `evidence`：入选证据。
- `trace_file`：trace JSON 路径。
- `timings`：耗时拆解。
- `messages`：原消息加 assistant 回答。
- `metadata`：包含 `agent_name`、`grounding_mode`、`llm_model`、`handoff_ready` 等。
- `errors`：仅失败时出现。

## CLI

默认问答入口现在会先进入 `brain_agent`，再由大脑 Agent 按需调用本地 RAG 子智能体：

```powershell
.\start_rag.ps1 --query 现在机器人行情怎么样
```

直接运行本地 RAG 子智能体：

```powershell
..\.venv\Scripts\python.exe -m rag_pipeline.agents.rag_agent --query 现在机器人行情怎么样
```

原始检索引擎调试仍可通过 `--json` 使用：

```powershell
.\start_rag.ps1 --json --query 现在机器人行情怎么样
```

## 多智能体用法

直接使用编译后的子图：

```python
from rag_pipeline.agents.rag_agent import create_rag_agent_graph

rag_agent = create_rag_agent_graph()
state = rag_agent.invoke({"query": "现在机器人行情怎么样"})
print(state["answer_text"])
```

或暴露为工具，让更上层 Supervisor 调用：

```python
from rag_pipeline.agents.rag_agent import create_rag_agent_tool

rag_tool = create_rag_agent_tool()
```

该工具适合在请求需要本地知识库、内部资料或可追溯行研证据时由 Supervisor 调用。
