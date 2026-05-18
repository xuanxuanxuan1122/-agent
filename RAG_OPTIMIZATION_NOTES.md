# RAG optimization notes

This file records which optimization suggestions are already useful for this
project, which ones have been implemented, and which ones should wait for a
database rebuild, a local reranker model, or a labeled eval set.

## Implemented in this pass

- `rag_pipeline/search/engine.py`
  - Added query latency breakdown: planning, embedding, sparse query build, retrieval, rerank, evidence selection, synthesis, and total time.
  - Added an embedding model cache for service mode via `RAG_KEEP_EMBEDDER_LOADED=1`; CLI still pays process startup cost.
  - Changed hierarchical retrieval to run parent-gated retrieval and flat child retrieval together, then merge.
  - Added weighted RRF controls:
    - `RAG_DENSE_RRF_WEIGHT`
    - `RAG_SPARSE_RRF_WEIGHT`
    - `RAG_FLAT_CHILD_RRF_WEIGHT`
    - `RAG_PARENT_RRF_WEIGHT`
  - Added optional local CrossEncoder rerank hooks:
    - `--enable-local-rerank`
    - `--local-rerank-model-path`
    - `--local-rerank-max-docs`
    - `--local-rerank-batch-size`
  - Added a final fact/definition gate to reduce weak table-like sparse hits in final results.

- `rag_pipeline/ingest/embedding_qdrant.py`
  - Added optional Chinese tokenizer support for sparse vectors with compatibility n-grams retained.
  - Added configurable HNSW build parameters:
    - `QDRANT_HNSW_M`, default `32`
    - `QDRANT_HNSW_EF_CONSTRUCT`, default `256`
  - Existing collection validation shows the current live collection is still `m=16`, `ef_construct=100`; rebuild or collection update is needed to apply the new defaults.

- `rag_pipeline/tools/benchmark.py`
  - Added labeled eval metrics: `MRR@10`, `nDCG@10`.
  - Added `task_type` grouped metrics.
  - Added average latency breakdown from `run_search`.

- `rag_pipeline/tools/validate_qdrant_store.py`
  - Added payload completeness sampling.
  - Added sparse sanity query check.

- `rag_pipeline/tools/audit_search_qdrant_defs.py`
  - Added duplicate top-level definition audit for `rag_pipeline/search/engine.py`.
  - Expanded the audit to also catch duplicate top-level constant-like assignments.
  - Cleaned the duplicate top-level defs and duplicate regex/constant assignments from `rag_pipeline/search/engine.py`.

## Confirmed useful, but needs a rebuild or model

- Better sparse retrieval
  - The collection already uses Qdrant sparse `IDF` modifier, so the immediate gap is token quality and sparse weighting.
  - Installing `jieba` and rebuilding the collection will make sparse vectors cleaner.
  - A full BM25 migration should be handled as a rebuild task, not a small search-only patch.

- Local reranker
  - `sentence_transformers` and `FlagEmbedding` are installed.
  - No local reranker model path was found on disk.
  - Recommended next model choices:
    - `bge-reranker-v2-m3` for multilingual/Chinese robustness.
    - `Qwen3-Reranker-0.6B` if you want Qwen-family consistency and have enough VRAM headroom.

- HNSW build quality
  - Live collection currently reports `m=16`, `ef_construct=100`.
  - New defaults are `m=32`, `ef_construct=256`.
  - Rebuilding is recommended before serious recall testing.

## Confirmed technical debt

The highest-risk duplicate-definition debt in `rag_pipeline/search/engine.py` has been removed, and the audit now passes.

Run:

```powershell
D:\pychram\RAG2\.venv\Scripts\python.exe -m rag_pipeline.tools.audit_search_qdrant_defs
```

Remaining structural debt is still the monolithic file layout. Recommended cleanup order:

1. Move shared text utilities out of `rag_pipeline/search/engine.py`.
2. Move active planner functions into `rag_pipeline/search/query_planning.py`.
3. Move Qdrant dense/sparse retrieval into `rag_pipeline/search/retrievers.py`.
4. Move scoring and rerank into `rag_pipeline/search/scoring.py`.
5. Move evidence selection into `rag_pipeline/search/evidence.py`.
6. Move answer rendering and CLI glue into smaller modules.
7. Keep `rag_pipeline/search/engine.py` as orchestration only.

## What not to prioritize yet

- LTR training should wait until there is a real labeled eval set with enough positive and negative examples.
- Intent classifier training should wait until there are at least a few hundred labeled questions.
- Full query HyDE should be introduced after the external planner API is stable, otherwise it can add latency without reliable recall gains.
