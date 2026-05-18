# Qdrant deployment

This project now defaults to text-first ingestion:

- `RAG_TEXT_SOURCE_PROFILE=clean_text`
- `RAG_EMBEDDING_REFINEMENT_MODE=off`

## 1. Start Qdrant

Run the commands below from `current_rag_pipeline/`.

```powershell
docker compose -f .\docker-compose.qdrant.yml up -d
```

Check service:

```powershell
Invoke-WebRequest http://127.0.0.1:6333/collections
```

## 2. Configure environment

所有项目级环境变量统一写入项目根目录的 `.env`。Qdrant、LLM planner、回答综合、回答审查、reflection 和 IQS 联网分析变量都在同一个文件中维护，不再使用其他 env 配置文件。

需要修改连接地址、模型名或 API Key 时，直接编辑：

```powershell
notepad .\.env
```

## 3. Build chunks

```powershell
..\.venv\Scripts\python.exe -B -m rag_pipeline.ingest.slicing
```

You can also run cleaning, chunking, embedding, and Qdrant sync as one pipeline:

```powershell
.\start_rag.ps1 ingest
```

If you already have `*.embedded.json` files and only want to sync them into Qdrant, use:

```powershell
.\start_rag.ps1 sync
```

## 4. Vectorize and push to Qdrant server

```powershell
..\.venv\Scripts\python.exe -B -m rag_pipeline.ingest.embedding_qdrant `
  --url $env:QDRANT_URL `
  --batch-size 16 `
  --embed-scope qdrant `
  --hnsw-m 32 `
  --hnsw-ef-construct 256
```

## 5. Search from Qdrant server

```powershell
..\.venv\Scripts\python.exe -B -m rag_pipeline.search.engine `
  --url $env:QDRANT_URL `
  "你的问题"
```

Or use the helper launcher, which fills in `QDRANT_URL` and `QDRANT_COLLECTION_NAME` automatically:

```powershell
.\start_rag.ps1 "your question"
```

Default search output is now concise Chinese. Add `--verbose` if you want the detailed retrieval diagnostics.

Useful low-latency / higher-recall search toggles:

```powershell
.\start_rag.ps1 "人工智能能引领下个产业革命浪潮吗？" `
  --parallel-hierarchical `
  --hnsw-ef 96 `
  --quantization-rescore
```

If you have a local cross-encoder reranker model, enable it like this:

```powershell
.\start_rag.ps1 "基金投资策略" `
  --enable-local-rerank `
  --local-rerank-model-path "D:\models\bge-reranker-v2-m3"
```

If you want to let an external LLM only do query planning, while retrieval/rerank/evidence selection still stay local, run:

```powershell
.\start_rag.ps1 "基金投资策略" `
  --enable-llm-planner `
  --llm-planner-url $env:RAG_LLM_PLANNER_URL `
  --llm-planner-api-key $env:RAG_LLM_PLANNER_API_KEY `
  --llm-planner-model $env:RAG_LLM_PLANNER_MODEL
```

If the external planner fails or returns invalid JSON, the script falls back to the local rule-based planner automatically.

If you want the external LLM to generate the final grounded answer, review the answer, and decide whether another retrieval hop is needed, run:

```powershell
.\start_rag.ps1 "宁德时代储能规模，以及它毛利率的变化原因" `
  --session-id demo-session `
  --enable-memory `
  --enable-contextualizer `
  --enable-llm-synthesis `
  --enable-answer-review `
  --enable-llm-answer-review `
  --enable-reflection `
  --enable-llm-reflection
```

When enabled, the system flow becomes:

1. rewrite the follow-up question into a standalone query
2. retrieve and rerank evidence from Qdrant
3. let the LLM generate a grounded answer with inline citations
4. let the LLM review that answer and either revise it or refuse it
5. reflect on evidence coverage and re-retrieve when the answer is still incomplete

## 6. Validate and benchmark

Validate payload completeness and sparse sanity recall:

```powershell
..\.venv\Scripts\python.exe -B -m rag_pipeline.tools.validate_qdrant_store `
  --url $env:QDRANT_URL `
  --collection rag_local_chunks `
  --sanity-query "人工智能 产业革命"
```

Audit duplicate definitions before refactoring `rag_pipeline.search.engine`:

```powershell
..\.venv\Scripts\python.exe -B -m rag_pipeline.tools.audit_search_qdrant_defs
```

Run benchmark and labeled eval:

```powershell
..\.venv\Scripts\python.exe -B -m rag_pipeline.tools.benchmark `
  --eval-file .\rag_eval_dataset.sample.jsonl `
  --eval-limit 20
```

## Notes

- Use Qdrant server mode for concurrent write/search. It avoids local file-lock conflicts from `qdrant_local`.
- Keep `embed-scope=qdrant` unless you explicitly need embeddings for every chunk in the JSON cache.
- If you deploy Qdrant remotely, just replace `QDRANT_URL` with your server address.
- The compatibility warning is disabled for the current local server because this repo is pinned to the running server on `127.0.0.1:6333`.
