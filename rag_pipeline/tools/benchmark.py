import argparse
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..ingest.embedding_qdrant import (
    DEFAULT_ATTN_IMPL,
    DEFAULT_BGE_M3_BATCH_SIZE,
    DEFAULT_BGE_M3_DEVICE,
    DEFAULT_BGE_M3_MODEL_PATH,
    DEFAULT_BGE_M3_PASSAGE_MAX_LENGTH,
    DEFAULT_BGE_M3_QUERY_MAX_LENGTH,
    DEFAULT_BGE_M3_USE_FP16,
    DEFAULT_COLLECTION,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_QDRANT_API_KEY,
    DEFAULT_QDRANT_PATH,
    DEFAULT_QDRANT_PREFER_GRPC,
    DEFAULT_QDRANT_URL,
    QwenLocalEmbeddingModel,
)
from ..search.engine import (
    DEFAULT_DENSE_VECTOR_NAME,
    build_arg_parser as build_search_arg_parser,
    build_client,
    build_qdrant_lock_message,
    build_search_params,
    is_qdrant_local_lock_error,
    probe_local_qdrant_url,
    resolve_local_qdrant_path,
    run_search,
    _require_qdrant_client,
)


def parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def pick_text(payload: Dict[str, Any]) -> str:
    for key in ("embedding_text", "retrieval_text", "text"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def build_query_from_text(text: str, max_len: int = 96) -> str:
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    for sep in ("。", "！", "？", ".", "!", "?", "\n"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    if len(text) > max_len:
        text = text[:max_len].strip()
    return text


def iter_chunk_payloads(chunk_root: Path, include_parents: bool) -> Iterable[Dict[str, Any]]:
    for chunk_file in sorted(chunk_root.rglob("*.chunks.json")):
        data = json.loads(chunk_file.read_text(encoding="utf-8"))
        for item in data.get("chunks", []):
            yield item
        if include_parents:
            for item in data.get("parent_chunks", []):
                yield item


def load_samples(chunk_root: Path, limit: int, include_parents: bool) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    queries: List[str] = []
    seen_texts: set[str] = set()
    seen_queries: set[str] = set()

    for payload in iter_chunk_payloads(chunk_root, include_parents=include_parents):
        text = pick_text(payload)
        if text and text not in seen_texts:
            seen_texts.add(text)
            texts.append(text)
        query = build_query_from_text(
            str(payload.get("summary_1line") or payload.get("section_title") or text)
        )
        if query and query not in seen_queries:
            seen_queries.add(query)
            queries.append(query)
        if len(texts) >= limit and len(queries) >= limit:
            break

    return texts[:limit], queries[:limit]


def load_eval_examples(eval_file: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with eval_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            item["query"] = str(item.get("query") or "").strip()
            item["gold_chunk_uids"] = [str(value).strip() for value in item.get("gold_chunk_uids", []) if str(value).strip()]
            item["should_refuse"] = bool(item.get("should_refuse", False))
            item["filters"] = item.get("filters", {}) if isinstance(item.get("filters", {}), dict) else {}
            if item["query"]:
                examples.append(item)
            if limit and len(examples) >= limit:
                break
    return examples


def build_search_eval_args(args: argparse.Namespace) -> SimpleNamespace:
    base = build_search_arg_parser().parse_args([])
    base.top_k = max(1, args.search_top_k)
    base.score_threshold = None
    base.db_path = args.db_path
    base.url = args.url
    base.api_key = args.api_key
    base.prefer_grpc = args.prefer_grpc
    base.collection = args.collection
    base.model_path = args.model_path
    base.device = args.device
    base.dtype = args.dtype
    base.attn_implementation = args.attn_implementation
    base.max_length = args.max_length
    base.preview_chars = 240
    base.query_variants = 4
    base.candidate_multiplier = 10
    base.max_per_document = 4
    base.enable_planner = True
    base.planner_hard_filters = False
    base.hnsw_ef = max(parse_int_list(args.hnsw_efs) or [96])
    base.exact = False
    rescore_options = parse_bool_options(args.rescore_options)
    base.quantization_rescore = rescore_options[0] if rescore_options else True
    base.quantization_oversampling = 1.0
    base.intent = "auto"
    base.retrieval_mode = "hierarchical"
    base.parallel_hierarchical = True
    base.enable_sparse_retrieval = True
    base.bge_m3_model_path = args.bge_m3_model_path
    base.bge_m3_device = args.bge_m3_device
    base.bge_m3_batch_size = args.bge_m3_batch_size
    base.bge_m3_query_max_length = args.bge_m3_query_max_length
    base.bge_m3_passage_max_length = args.bge_m3_passage_max_length
    base.bge_m3_use_fp16 = args.bge_m3_use_fp16
    base.enable_bge_dense_retrieval = args.enable_bge_dense_retrieval
    base.enable_bge_sparse_retrieval = args.enable_bge_sparse_retrieval
    base.enable_api_rerank = False
    base.rerank_provider = "cohere"
    base.rerank_url = ""
    base.rerank_api_key = ""
    base.rerank_model = ""
    base.rerank_top_n = 0
    base.rerank_max_docs = 0
    base.rerank_max_chars_per_doc = 0
    base.rerank_timeout = 0.0
    base.answer_mode = "grounded"
    base.evidence_top_k = max(5, min(12, args.search_top_k))
    base.min_evidence = 2
    base.min_evidence_score = 0.55
    base.max_answer_claims = 4
    base.core_evidence_top_k = 4
    base.support_evidence_top_k = 12
    base.enable_llm_synthesis = False
    base.llm_synthesis_provider = "openai_compatible"
    base.llm_synthesis_url = ""
    base.llm_synthesis_api_key = ""
    base.llm_synthesis_model = ""
    base.llm_synthesis_timeout = 30.0
    base.enable_answer_review = False
    base.enable_llm_answer_review = False
    base.llm_answer_review_provider = "openai_compatible"
    base.llm_answer_review_url = ""
    base.llm_answer_review_api_key = ""
    base.llm_answer_review_model = ""
    base.llm_answer_review_timeout = 30.0
    base.enable_reflection = False
    base.reflection_max_hops = 1
    base.reflection_overlap_threshold = 0.8
    base.enable_llm_reflection = False
    base.llm_reflection_provider = "openai_compatible"
    base.llm_reflection_url = ""
    base.llm_reflection_api_key = ""
    base.llm_reflection_model = ""
    base.llm_reflection_timeout = 30.0
    base.session_id = ""
    base.enable_memory = False
    base.enable_contextualizer = False
    base.memory_store_dir = ""
    base.memory_max_turns = 10
    base.context_history_turns = 4
    base.enable_trace = False
    base.trace_dir = ""
    base.trace_top_k = 10
    base.source_file = ""
    base.doc_title = ""
    base.chunk_uid = ""
    base.chunk_type = ""
    base.chunk_level = ""
    base.disable_instruction = False
    base.no_embedder_cache = False
    base.json = False
    base.query = ""
    return base


def overlap_ratio(expected: Sequence[str], observed: Sequence[str]) -> float:
    expected_set = {value for value in expected if value}
    if not expected_set:
        return 0.0
    observed_set = {value for value in observed if value}
    return len(expected_set & observed_set) / max(len(expected_set), 1)


def reciprocal_rank(expected: Sequence[str], observed: Sequence[str], top_k: int = 10) -> float:
    expected_set = {value for value in expected if value}
    if not expected_set:
        return 0.0
    for rank, uid in enumerate(observed[:top_k], start=1):
        if uid in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(expected: Sequence[str], observed: Sequence[str], top_k: int = 10) -> float:
    expected_set = {value for value in expected if value}
    if not expected_set:
        return 0.0
    dcg = 0.0
    for rank, uid in enumerate(observed[:top_k], start=1):
        if uid in expected_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(expected_set), top_k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def average_dicts(values: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not values:
        return {}
    keys = sorted({key for item in values for key in item})
    return {key: statistics.mean(float(item.get(key, 0.0) or 0.0) for item in values) for key in keys}


def evaluate_labeled_queries(eval_examples: Sequence[Dict[str, Any]], search_args: SimpleNamespace) -> Dict[str, Any]:
    if not eval_examples:
        return {}

    metrics = {
        "count": 0,
        "retrieval_hit_rate": 0.0,
        "retrieval_recall": 0.0,
        "evidence_hit_rate": 0.0,
        "evidence_recall": 0.0,
        "mrr_at_10": 0.0,
        "ndcg_at_10": 0.0,
        "refusal_accuracy": 0.0,
        "answerable_success_rate": 0.0,
    }
    per_query: List[Dict[str, Any]] = []
    grouped: Dict[str, Dict[str, Any]] = {}
    latency_records: List[Dict[str, float]] = []

    for example in eval_examples:
        runtime_args = SimpleNamespace(**vars(search_args))
        runtime_args.source_file = str(example.get("filters", {}).get("source_file", "") or "")
        runtime_args.doc_title = str(example.get("filters", {}).get("doc_title", "") or "")
        runtime_args.chunk_uid = str(example.get("filters", {}).get("chunk_uid", "") or "")
        runtime_args.chunk_type = str(example.get("filters", {}).get("chunk_type", "") or "")
        runtime_args.chunk_level = str(example.get("filters", {}).get("chunk_level", "") or "")

        output = run_search(runtime_args, example["query"])
        result_uids = [str(item.get("chunk_uid") or "") for item in output.get("results", [])]
        evidence_uids = [str(item.get("chunk_uid") or "") for item in output.get("evidence", [])]
        gold_chunk_uids = example.get("gold_chunk_uids", [])
        answer = output.get("answer") or {}
        plan = output.get("plan") or {}
        task_type = str(example.get("task_type") or plan.get("task_type") or "unknown")
        timings = {key: float(value or 0.0) for key, value in (output.get("timings") or {}).items()}
        latency_records.append(timings)
        refused = str(answer.get("status") or "") == "insufficient_evidence"
        should_refuse = bool(example.get("should_refuse", False))

        retrieval_hit = 1.0 if any(uid in set(gold_chunk_uids) for uid in result_uids) else 0.0
        evidence_hit = 1.0 if any(uid in set(gold_chunk_uids) for uid in evidence_uids) else 0.0
        retrieval_recall = overlap_ratio(gold_chunk_uids, result_uids)
        evidence_recall = overlap_ratio(gold_chunk_uids, evidence_uids)
        mrr_at_10 = reciprocal_rank(gold_chunk_uids, result_uids, top_k=10)
        ndcg10 = ndcg_at_k(gold_chunk_uids, result_uids, top_k=10)
        refusal_correct = 1.0 if refused == should_refuse else 0.0
        answerable_success = 1.0 if (not should_refuse and not refused and retrieval_hit > 0.0) else 0.0

        metrics["count"] += 1
        metrics["retrieval_hit_rate"] += retrieval_hit
        metrics["retrieval_recall"] += retrieval_recall
        metrics["evidence_hit_rate"] += evidence_hit
        metrics["evidence_recall"] += evidence_recall
        metrics["mrr_at_10"] += mrr_at_10
        metrics["ndcg_at_10"] += ndcg10
        metrics["refusal_accuracy"] += refusal_correct
        metrics["answerable_success_rate"] += answerable_success
        bucket = grouped.setdefault(
            task_type,
            {
                "count": 0,
                "retrieval_hit_rate": 0.0,
                "retrieval_recall": 0.0,
                "evidence_hit_rate": 0.0,
                "evidence_recall": 0.0,
                "mrr_at_10": 0.0,
                "ndcg_at_10": 0.0,
                "latencies": [],
            },
        )
        bucket["count"] += 1
        bucket["retrieval_hit_rate"] += retrieval_hit
        bucket["retrieval_recall"] += retrieval_recall
        bucket["evidence_hit_rate"] += evidence_hit
        bucket["evidence_recall"] += evidence_recall
        bucket["mrr_at_10"] += mrr_at_10
        bucket["ndcg_at_10"] += ndcg10
        bucket["latencies"].append(timings)
        per_query.append(
            {
                "query": example["query"],
                "task_type": task_type,
                "should_refuse": should_refuse,
                "predicted_status": answer.get("status", ""),
                "retrieval_hit": retrieval_hit,
                "retrieval_recall": round(retrieval_recall, 4),
                "evidence_hit": evidence_hit,
                "evidence_recall": round(evidence_recall, 4),
                "mrr_at_10": round(mrr_at_10, 4),
                "ndcg_at_10": round(ndcg10, 4),
                "timings": timings,
                "result_chunk_uids": result_uids,
                "evidence_chunk_uids": evidence_uids,
                "gold_chunk_uids": gold_chunk_uids,
            }
        )

    count = max(1, metrics["count"])
    for key in ("retrieval_hit_rate", "retrieval_recall", "evidence_hit_rate", "evidence_recall", "mrr_at_10", "ndcg_at_10", "refusal_accuracy", "answerable_success_rate"):
        metrics[key] = metrics[key] / count
    task_groups: Dict[str, Any] = {}
    for task_type, bucket in grouped.items():
        task_count = max(1, int(bucket["count"]))
        task_groups[task_type] = {
            "count": bucket["count"],
            "retrieval_hit_rate": bucket["retrieval_hit_rate"] / task_count,
            "retrieval_recall": bucket["retrieval_recall"] / task_count,
            "evidence_hit_rate": bucket["evidence_hit_rate"] / task_count,
            "evidence_recall": bucket["evidence_recall"] / task_count,
            "mrr_at_10": bucket["mrr_at_10"] / task_count,
            "ndcg_at_10": bucket["ndcg_at_10"] / task_count,
            "avg_timings_ms": average_dicts(bucket["latencies"]),
        }
    metrics["task_groups"] = task_groups
    metrics["avg_timings_ms"] = average_dicts(latency_records)
    metrics["queries"] = per_query
    return metrics


def percentile_ms(values_ms: Sequence[float], percentile: float) -> float:
    if not values_ms:
        return 0.0
    ordered = sorted(values_ms)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * percentile) - 1))
    return float(ordered[index])


def benchmark_embedding(
    embedder: QwenLocalEmbeddingModel,
    texts: Sequence[str],
    batch_sizes: Sequence[int],
    repeats: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    warmup_batch = list(texts[: min(len(texts), 4)])
    if warmup_batch:
        embedder.encode(warmup_batch, batch_size=min(4, max(batch_sizes or [4])))

    for batch_size in batch_sizes:
        latencies_ms: List[float] = []
        for _ in range(max(1, repeats)):
            started = time.perf_counter()
            embedder.encode(list(texts), batch_size=batch_size)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
        avg_ms = statistics.mean(latencies_ms)
        results.append(
            {
                "batch_size": batch_size,
                "avg_ms": avg_ms,
                "p95_ms": percentile_ms(latencies_ms, 0.95),
                "texts_per_sec": (len(texts) * 1000.0 / avg_ms) if avg_ms > 0 else 0.0,
            }
        )
    return results


def query_dense(
    client: Any,
    collection: str,
    query_vector: Sequence[float],
    search_params: Any,
    limit: int,
    qmodels: Any,
    child_only: bool,
) -> Tuple[float, List[str]]:
    query_filter = None
    if child_only:
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="chunk_level", match=qmodels.MatchValue(value="child"))]
        )
    started = time.perf_counter()
    response = client.query_points(
        collection_name=collection,
        query=list(query_vector),
        using=DEFAULT_DENSE_VECTOR_NAME,
        query_filter=query_filter,
        search_params=search_params,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ids = [str(point.payload.get("chunk_uid") or point.id) for point in response.points]
    return elapsed_ms, ids


def overlap_at_k(baseline: Sequence[str], candidate: Sequence[str], top_k: int) -> float:
    baseline_set = {value for value in baseline[:top_k] if value}
    if not baseline_set:
        return 0.0
    candidate_set = {value for value in candidate[:top_k] if value}
    return len(baseline_set & candidate_set) / max(len(baseline_set), 1)


def benchmark_search(
    client: Any,
    collection: str,
    embedder: QwenLocalEmbeddingModel,
    queries: Sequence[str],
    hnsw_efs: Sequence[int],
    rescore_options: Sequence[bool],
    top_k: int,
    child_only: bool,
) -> List[Dict[str, Any]]:
    _, qmodels = _require_qdrant_client()
    query_vectors = embedder.encode(list(queries), batch_size=min(16, max(1, len(queries))))
    baseline_params = build_search_params(
        qmodels=qmodels,
        hnsw_ef=max(hnsw_efs) if hnsw_efs else 128,
        exact=True,
        quantization_rescore=True,
        quantization_oversampling=1.0,
    )
    baselines: List[List[str]] = []
    for vector in query_vectors:
        _, ids = query_dense(
            client=client,
            collection=collection,
            query_vector=vector,
            search_params=baseline_params,
            limit=top_k,
            qmodels=qmodels,
            child_only=child_only,
        )
        baselines.append(ids)

    results: List[Dict[str, Any]] = []
    for hnsw_ef in hnsw_efs:
        for rescore in rescore_options:
            search_params = build_search_params(
                qmodels=qmodels,
                hnsw_ef=hnsw_ef,
                exact=False,
                quantization_rescore=rescore,
                quantization_oversampling=1.0,
            )
            latencies_ms: List[float] = []
            overlaps: List[float] = []
            for baseline_ids, vector in zip(baselines, query_vectors):
                elapsed_ms, ids = query_dense(
                    client=client,
                    collection=collection,
                    query_vector=vector,
                    search_params=search_params,
                    limit=top_k,
                    qmodels=qmodels,
                    child_only=child_only,
                )
                latencies_ms.append(elapsed_ms)
                overlaps.append(overlap_at_k(baseline_ids, ids, top_k))
            results.append(
                {
                    "hnsw_ef": hnsw_ef,
                    "rescore": rescore,
                    "avg_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
                    "p95_ms": percentile_ms(latencies_ms, 0.95),
                    "overlap_at_k": statistics.mean(overlaps) if overlaps else 0.0,
                }
            )
    return results


def choose_embedding_recommendation(results: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not results:
        return None
    return max(results, key=lambda item: (item["texts_per_sec"], -item["avg_ms"]))


def choose_search_recommendation(results: Sequence[Dict[str, Any]], min_overlap: float) -> Dict[str, Any] | None:
    if not results:
        return None
    eligible = [item for item in results if item["overlap_at_k"] >= min_overlap]
    if eligible:
        return min(eligible, key=lambda item: (item["p95_ms"], item["avg_ms"], -item["overlap_at_k"]))
    return max(results, key=lambda item: (item["overlap_at_k"], -item["p95_ms"]))


def print_embedding_results(results: Sequence[Dict[str, Any]], recommendation: Dict[str, Any] | None) -> None:
    print("\n[Embedding Benchmark]")
    for item in results:
        print(
            f"batch_size={item['batch_size']:>3} "
            f"avg_ms={item['avg_ms']:.1f} "
            f"p95_ms={item['p95_ms']:.1f} "
            f"texts_per_sec={item['texts_per_sec']:.2f}"
        )
    if recommendation:
        print(
            f"[RECOMMEND] embedding batch_size={recommendation['batch_size']} "
            f"(texts_per_sec={recommendation['texts_per_sec']:.2f}, avg_ms={recommendation['avg_ms']:.1f})"
        )


def print_search_results(results: Sequence[Dict[str, Any]], recommendation: Dict[str, Any] | None) -> None:
    print("\n[Search Benchmark]")
    for item in results:
        print(
            f"hnsw_ef={item['hnsw_ef']:>3} "
            f"rescore={str(item['rescore']):<5} "
            f"avg_ms={item['avg_ms']:.1f} "
            f"p95_ms={item['p95_ms']:.1f} "
            f"overlap_at_k={item['overlap_at_k']:.3f}"
        )
    if recommendation:
        print(
            f"[RECOMMEND] search hnsw_ef={recommendation['hnsw_ef']} "
            f"rescore={recommendation['rescore']} "
            f"(p95_ms={recommendation['p95_ms']:.1f}, overlap_at_k={recommendation['overlap_at_k']:.3f})"
        )


def print_eval_results(results: Dict[str, Any]) -> None:
    if not results:
        return
    print("\n[Labeled Eval]")
    print(
        f"count={results['count']} "
        f"retrieval_hit_rate={results['retrieval_hit_rate']:.3f} "
        f"retrieval_recall={results['retrieval_recall']:.3f} "
        f"evidence_hit_rate={results['evidence_hit_rate']:.3f} "
        f"evidence_recall={results['evidence_recall']:.3f} "
        f"mrr@10={results['mrr_at_10']:.3f} "
        f"ndcg@10={results['ndcg_at_10']:.3f} "
        f"refusal_accuracy={results['refusal_accuracy']:.3f} "
        f"answerable_success_rate={results['answerable_success_rate']:.3f}"
    )
    if results.get("avg_timings_ms"):
        print(f"avg_timings_ms={json.dumps(results['avg_timings_ms'], ensure_ascii=False)}")
    if results.get("task_groups"):
        print("[By task_type]")
        for task_type, item in sorted(results["task_groups"].items()):
            print(
                f"{task_type}: count={item['count']} "
                f"hit={item['retrieval_hit_rate']:.3f} "
                f"recall={item['retrieval_recall']:.3f} "
                f"mrr@10={item['mrr_at_10']:.3f} "
                f"ndcg@10={item['ndcg_at_10']:.3f} "
                f"avg_timings_ms={json.dumps(item.get('avg_timings_ms', {}), ensure_ascii=False)}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark embedding throughput and Qdrant dense-search latency.")
    parser.add_argument("--chunk-root", default="rag_chunks_store", help="Directory containing *.chunks.json files.")
    parser.add_argument("--sample-size", type=int, default=96, help="How many sample passages / queries to benchmark.")
    parser.add_argument("--include-parents", action="store_true", help="Include parent chunks in the benchmark samples.")
    parser.add_argument("--embedding-batch-sizes", default="8,16,24,32", help="Comma-separated embedding batch sizes to test.")
    parser.add_argument("--embedding-repeats", type=int, default=2, help="How many times to repeat each embedding batch-size test.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Local embedding model path.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Embedding device, e.g. auto/cpu/cuda:0.")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, help="Embedding dtype.")
    parser.add_argument("--attn-implementation", default=DEFAULT_ATTN_IMPL, help="Transformers attention implementation.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Tokenizer max length.")
    parser.add_argument("--bge-m3-model-path", default=str(DEFAULT_BGE_M3_MODEL_PATH), help="Local BGE-M3 model path used by search eval runs.")
    parser.add_argument("--bge-m3-device", default=DEFAULT_BGE_M3_DEVICE, help="BGE-M3 device for search eval runs.")
    parser.add_argument("--bge-m3-batch-size", type=int, default=DEFAULT_BGE_M3_BATCH_SIZE, help="BGE-M3 query batch size for search eval runs.")
    parser.add_argument("--bge-m3-query-max-length", type=int, default=DEFAULT_BGE_M3_QUERY_MAX_LENGTH, help="BGE-M3 query tokenizer max length.")
    parser.add_argument("--bge-m3-passage-max-length", type=int, default=DEFAULT_BGE_M3_PASSAGE_MAX_LENGTH, help="BGE-M3 passage tokenizer max length.")
    parser.add_argument("--bge-m3-use-fp16", dest="bge_m3_use_fp16", action="store_true", default=DEFAULT_BGE_M3_USE_FP16, help="Enable fp16 when loading BGE-M3 on CUDA.")
    parser.add_argument("--no-bge-m3-use-fp16", dest="bge_m3_use_fp16", action="store_false", help="Disable fp16 for BGE-M3.")
    parser.add_argument("--enable-bge-dense-retrieval", dest="enable_bge_dense_retrieval", action="store_true", default=bool(DEFAULT_BGE_M3_MODEL_PATH), help="Enable BGE-M3 dense retrieval during labeled eval runs.")
    parser.add_argument("--disable-bge-dense-retrieval", dest="enable_bge_dense_retrieval", action="store_false", help="Disable BGE-M3 dense retrieval during labeled eval runs.")
    parser.add_argument("--enable-bge-sparse-retrieval", dest="enable_bge_sparse_retrieval", action="store_true", default=bool(DEFAULT_BGE_M3_MODEL_PATH), help="Enable BGE-M3 sparse retrieval during labeled eval runs.")
    parser.add_argument("--disable-bge-sparse-retrieval", dest="enable_bge_sparse_retrieval", action="store_false", help="Disable BGE-M3 sparse retrieval during labeled eval runs.")
    parser.add_argument("--skip-search", action="store_true", help="Only benchmark embedding throughput.")
    parser.add_argument("--db-path", default=str(DEFAULT_QDRANT_PATH), help="Local Qdrant storage directory.")
    parser.add_argument("--url", default=DEFAULT_QDRANT_URL, help="Qdrant server URL.")
    parser.add_argument("--api-key", default=DEFAULT_QDRANT_API_KEY, help="Optional Qdrant API key.")
    parser.add_argument("--prefer-grpc", action="store_true", default=DEFAULT_QDRANT_PREFER_GRPC, help="Prefer gRPC when using a Qdrant server URL.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--search-top-k", type=int, default=8, help="Top-k used for search overlap benchmarking.")
    parser.add_argument("--hnsw-efs", default="64,96,128", help="Comma-separated HNSW ef values to test.")
    parser.add_argument("--rescore-options", default="true,false", help="Comma-separated rescore options to test.")
    parser.add_argument("--search-child-only", dest="search_child_only", action="store_true", default=True, help="Benchmark dense search against child chunks only.")
    parser.add_argument("--search-all-levels", dest="search_child_only", action="store_false", help="Benchmark dense search against all chunk levels.")
    parser.add_argument("--min-overlap", type=float, default=0.90, help="Minimum overlap@k target when choosing the recommended search config.")
    parser.add_argument("--eval-file", default="", help="Optional JSONL file with labeled queries and gold_chunk_uids.")
    parser.add_argument("--eval-limit", type=int, default=0, help="Optional limit for labeled eval examples.")
    return parser


def parse_bool_options(raw: str) -> List[bool]:
    values: List[bool] = []
    for part in str(raw or "").split(","):
        token = part.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            values.append(True)
        elif token in {"0", "false", "no", "off"}:
            values.append(False)
    return values or [True, False]


def main() -> int:
    args = build_arg_parser().parse_args()
    chunk_root = Path(args.chunk_root)
    texts, queries = load_samples(chunk_root, limit=args.sample_size, include_parents=args.include_parents)
    if not texts:
        raise RuntimeError(f"No benchmark samples found under: {chunk_root}")

    batch_sizes = parse_int_list(args.embedding_batch_sizes)
    hnsw_efs = parse_int_list(args.hnsw_efs)
    rescore_options = parse_bool_options(args.rescore_options)

    embedder = QwenLocalEmbeddingModel(
        model_name_or_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
    )
    try:
        embedding_results = benchmark_embedding(
            embedder=embedder,
            texts=texts,
            batch_sizes=batch_sizes,
            repeats=args.embedding_repeats,
        )
        print_embedding_results(
            embedding_results,
            choose_embedding_recommendation(embedding_results),
        )

        if args.skip_search:
            return 0

        try:
            resolved_url = args.url or probe_local_qdrant_url(args.collection)
            resolved_db_path = args.db_path if resolved_url else resolve_local_qdrant_path(args.db_path, args.collection)
            client = build_client(resolved_db_path, resolved_url, args.api_key, args.prefer_grpc)
        except RuntimeError as exc:
            if not resolved_url and is_qdrant_local_lock_error(exc):
                raise RuntimeError(build_qdrant_lock_message(resolved_db_path)) from exc
            raise
        try:
            if not client.collection_exists(args.collection):
                print(f"\n[WARN] Qdrant collection not found, skipping search benchmark: {args.collection}")
                return 0
            search_results = benchmark_search(
                client=client,
                collection=args.collection,
                embedder=embedder,
                queries=queries,
                hnsw_efs=hnsw_efs,
                rescore_options=rescore_options,
                top_k=args.search_top_k,
                child_only=args.search_child_only,
            )
            print_search_results(
                search_results,
                choose_search_recommendation(search_results, min_overlap=args.min_overlap),
            )

            if args.eval_file:
                eval_file = Path(args.eval_file)
                if not eval_file.exists():
                    raise RuntimeError(f"Labeled eval file not found: {eval_file}")
                eval_examples = load_eval_examples(eval_file, limit=args.eval_limit or None)
                eval_results = evaluate_labeled_queries(
                    eval_examples=eval_examples,
                    search_args=build_search_eval_args(args),
                )
                print_eval_results(eval_results)
        finally:
            client.close()
    finally:
        embedder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
