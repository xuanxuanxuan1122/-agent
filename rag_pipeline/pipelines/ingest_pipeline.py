import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from ..ingest import embedding_qdrant as embedding
from ..ingest import slicing


def probe_qdrant_server() -> str:
    for url in ("http://127.0.0.1:6333", "http://localhost:6333"):
        try:
            with urllib.request.urlopen(f"{url}/collections", timeout=1.5):
                return url
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    return ""


def resolve_qdrant_url(raw_url: str) -> str:
    if raw_url:
        return raw_url
    return probe_qdrant_server()


def run_slicing_stage(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input_path)
    chunk_output_dir = Path(args.chunk_output_dir)

    slicing.INPUT_PATH = input_path
    slicing.OUTPUT_DIR = chunk_output_dir
    slicing.FORCE_REPROCESS = bool(args.force or args.force_slicing)
    slicing.FAST_MODE = bool(args.fast_slicing)

    input_files = slicing.iter_input_files(input_path)
    if not input_files:
        raise FileNotFoundError(f"No supported .txt files found under: {input_path}")

    slicing.preflight_runtime_checks(input_files)
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0
    index: List[Dict[str, Any]] = []

    for input_file in input_files:
        output_file = slicing.build_output_file_path(input_file, input_path, chunk_output_dir)
        if output_file.exists() and not slicing.FORCE_REPROCESS:
            print(f"[SKIP] Already sliced: {input_file}")
            skipped += 1
            index.append({"source_file": str(input_file), "output_file": str(output_file), "status": "skipped"})
            continue

        try:
            final_chunks, parent_chunks = slicing.process_one_input_file_with_hierarchy(input_file)
            slicing.write_chunks_file(output_file, input_file, final_chunks, parent_chunks)
            if args.preview_top_k > 0:
                slicing.print_preview(final_chunks, top_k=args.preview_top_k)
            processed += 1
            index.append(
                {
                    "source_file": str(input_file),
                    "output_file": str(output_file),
                    "status": "processed",
                    "chunk_count": len(final_chunks),
                    "parent_chunk_count": len(parent_chunks),
                }
            )
        except Exception as exc:
            failed += 1
            print(f"[ERROR] Failed to slice {input_file}: {exc}")
            if "CUDA device was requested" in str(exc) or "Missing dependency" in str(exc):
                raise
            index.append({"source_file": str(input_file), "output_file": str(output_file), "status": "failed", "error": str(exc)})

    index_path = chunk_output_dir / "_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "index_file": str(index_path),
        "chunk_output_dir": str(chunk_output_dir),
    }


def run_embedding_stage(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.chunk_output_dir)
    output_dir = Path(args.embedding_output_dir)
    chunk_files = embedding.iter_chunk_files(input_path)
    if not chunk_files:
        raise FileNotFoundError(f"No .chunks.json files found under: {input_path}")

    store_to_qdrant = not args.no_qdrant
    qdrant_url = resolve_qdrant_url(args.url) if store_to_qdrant else ""
    embedding.preflight_runtime_checks(
        chunk_files,
        args.model_path,
        args.device,
        store_to_qdrant,
        bge_m3_model_path=args.bge_m3_model_path,
    )

    embedder = embedding.QwenLocalEmbeddingModel(
        model_name_or_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
    )
    bge_helper = embedding.BgeM3IngestHelper(
        model_path=args.bge_m3_model_path,
        device=args.bge_m3_device,
        batch_size=args.bge_m3_batch_size,
        query_max_length=args.bge_m3_query_max_length,
        passage_max_length=args.bge_m3_passage_max_length,
        use_fp16=args.bge_m3_use_fp16,
    )

    store = None
    if store_to_qdrant:
        store = embedding.QdrantLocalVectorStore(
            db_path=args.db_path,
            collection_name=args.collection,
            drop_if_exists=args.drop_if_exists,
            url=qdrant_url,
            api_key=args.api_key,
            prefer_grpc=args.prefer_grpc,
            dense_on_disk=args.dense_on_disk,
            hnsw_on_disk=args.hnsw_on_disk,
            on_disk_payload=args.on_disk_payload,
            enable_scalar_quantization=args.enable_scalar_quantization,
            scalar_quantization_always_ram=args.scalar_quantization_always_ram,
            upsert_wait=args.qdrant_upsert_wait,
            enable_bge_dense=bool(bge_helper.enabled),
            enable_bge_sparse=bool(bge_helper.enabled),
        )

    processed = 0
    reused = 0
    skipped = 0
    failed = 0
    index: List[Dict[str, Any]] = []

    try:
        for chunk_file in chunk_files:
            try:
                summary = embedding.process_one_chunk_file(
                    chunk_file=chunk_file,
                    input_root=input_path,
                    output_root=output_dir,
                    embedder=embedder,
                    store=store,
                    bge_helper=bge_helper,
                    batch_size=args.batch_size,
                    max_retries=args.max_retries,
                    initial_delay=args.initial_delay,
                    request_interval=args.request_interval,
                    write_json=args.write_json,
                    force=bool(args.force or args.force_embedding),
                    reupsert_existing=args.reupsert_existing,
                    enrich_metadata=args.enrich_metadata,
                    embed_scope=args.embed_scope,
                    preview_top_k=args.preview_top_k,
                )
                if summary["status"] == "reused":
                    reused += 1
                elif summary["status"] == "skipped":
                    skipped += 1
                else:
                    processed += 1
                index.append(summary)
            except Exception as exc:
                failed += 1
                print(f"[ERROR] Failed to vectorize {chunk_file}: {exc}")
                index.append({"source_chunk_file": str(chunk_file), "status": "failed", "error": str(exc)})
    finally:
        if store is not None:
            store.close()
        embedder.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "_embedding_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "processed": processed,
        "reused": reused,
        "skipped": skipped,
        "failed": failed,
        "index_file": str(index_path),
        "qdrant_url": qdrant_url,
        "collection": args.collection,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run txt cleaning, chunking, embedding, and Qdrant sync in one pipeline.")
    parser.add_argument("--input-path", default=str(slicing.INPUT_PATH), help="Raw .txt file or directory.")
    parser.add_argument("--chunk-output-dir", default=str(slicing.OUTPUT_DIR), help="Directory for *.chunks.json files.")
    parser.add_argument("--embedding-output-dir", default=str(embedding.DEFAULT_EMBED_OUTPUT), help="Directory for *.embedded.json and embedding index files.")
    parser.add_argument("--skip-slicing", action="store_true", help="Skip text cleaning/chunking and only vectorize existing chunks.")
    parser.add_argument("--skip-embedding", action="store_true", help="Only clean/chunk text and skip vectorization/Qdrant sync.")
    parser.add_argument("--force", action="store_true", help="Force both slicing and embedding stages to recompute.")
    parser.add_argument("--force-slicing", action="store_true", help="Force the slicing stage to regenerate chunks.")
    parser.add_argument("--force-embedding", action="store_true", help="Force the embedding stage to regenerate vectors.")
    parser.add_argument("--fast-slicing", action="store_true", help="Enable FAST_MODE in the slicing stage.")
    parser.add_argument("--preview-top-k", type=int, default=1, help="Preview chunks per file in each stage; set 0 to disable.")

    parser.add_argument("--model-path", default=embedding.DEFAULT_MODEL_PATH, help="Local embedding model path.")
    parser.add_argument("--device", default=embedding.DEFAULT_DEVICE, help="Embedding device, e.g. auto/cpu/cuda:0.")
    parser.add_argument("--dtype", default=embedding.DEFAULT_DTYPE, help="Embedding dtype.")
    parser.add_argument("--attn-implementation", default=embedding.DEFAULT_ATTN_IMPL, help="Transformers attention implementation.")
    parser.add_argument("--max-length", type=int, default=embedding.DEFAULT_MAX_LENGTH, help="Tokenizer max length.")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--max-retries", type=int, default=embedding.DEFAULT_MAX_RETRIES, help="Max retries for a failed embedding batch.")
    parser.add_argument("--initial-delay", type=float, default=embedding.DEFAULT_INITIAL_DELAY, help="Initial retry delay in seconds.")
    parser.add_argument("--request-interval", type=float, default=embedding.DEFAULT_REQUEST_INTERVAL, help="Delay between embedding batches in seconds.")

    parser.add_argument("--url", default=embedding.DEFAULT_QDRANT_URL, help="Qdrant server URL. Defaults to local server auto-detection.")
    parser.add_argument("--api-key", default=embedding.DEFAULT_QDRANT_API_KEY, help="Optional Qdrant API key.")
    parser.add_argument("--prefer-grpc", action="store_true", default=embedding.DEFAULT_QDRANT_PREFER_GRPC, help="Prefer gRPC for server mode.")
    parser.add_argument("--db-path", default=str(embedding.DEFAULT_QDRANT_PATH), help="Fallback local Qdrant path.")
    parser.add_argument("--collection", default=embedding.DEFAULT_COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--drop-if-exists", action="store_true", default=embedding.DEFAULT_DROP_IF_EXISTS, help="Drop the collection once before first write.")
    parser.add_argument("--reupsert-existing", action="store_true", default=embedding.DEFAULT_REUPSERT_EXISTING, help="Re-upsert embedded JSON that already matches the source chunks.")
    parser.add_argument("--write-json", dest="write_json", action="store_true", default=embedding.DEFAULT_WRITE_EMBEDDED_JSON, help="Write *.embedded.json files.")
    parser.add_argument("--no-write-json", dest="write_json", action="store_false", help="Skip writing *.embedded.json files.")
    parser.add_argument("--no-qdrant", action="store_true", help="Skip Qdrant sync and only write embeddings when enabled.")
    parser.add_argument("--embed-scope", choices=["qdrant", "all"], default=embedding.DEFAULT_EMBED_SCOPE, help="Vectorize Qdrant-eligible chunks or all chunks.")
    parser.add_argument("--enrich-metadata", dest="enrich_metadata", action="store_true", default=embedding.DEFAULT_ENABLE_METADATA_ENRICH, help="Enrich chunks before vectorization.")
    parser.add_argument("--no-enrich-metadata", dest="enrich_metadata", action="store_false", help="Disable metadata enrichment.")
    parser.add_argument("--bge-m3-model-path", default=embedding.DEFAULT_BGE_M3_MODEL_PATH, help="Local BGE-M3 model path used to enrich chunks with dense+sparse features.")
    parser.add_argument("--bge-m3-device", default=embedding.DEFAULT_BGE_M3_DEVICE, help="BGE-M3 device, e.g. auto/cpu/cuda:0.")
    parser.add_argument("--bge-m3-batch-size", type=int, default=embedding.DEFAULT_BGE_M3_BATCH_SIZE, help="BGE-M3 batch size.")
    parser.add_argument("--bge-m3-query-max-length", type=int, default=embedding.DEFAULT_BGE_M3_QUERY_MAX_LENGTH, help="BGE-M3 query tokenizer max length.")
    parser.add_argument("--bge-m3-passage-max-length", type=int, default=embedding.DEFAULT_BGE_M3_PASSAGE_MAX_LENGTH, help="BGE-M3 passage tokenizer max length.")
    parser.add_argument("--bge-m3-use-fp16", dest="bge_m3_use_fp16", action="store_true", default=embedding.DEFAULT_BGE_M3_USE_FP16, help="Enable fp16 when loading BGE-M3 on CUDA.")
    parser.add_argument("--no-bge-m3-use-fp16", dest="bge_m3_use_fp16", action="store_false", help="Disable fp16 for BGE-M3.")

    parser.add_argument("--dense-on-disk", dest="dense_on_disk", action="store_true", default=embedding.DEFAULT_QDRANT_DENSE_ON_DISK, help="Store dense vectors on disk.")
    parser.add_argument("--no-dense-on-disk", dest="dense_on_disk", action="store_false", help="Keep dense vectors in RAM.")
    parser.add_argument("--hnsw-on-disk", dest="hnsw_on_disk", action="store_true", default=embedding.DEFAULT_QDRANT_HNSW_ON_DISK, help="Store HNSW graph on disk.")
    parser.add_argument("--no-hnsw-on-disk", dest="hnsw_on_disk", action="store_false", help="Keep HNSW graph in RAM.")
    parser.add_argument("--on-disk-payload", dest="on_disk_payload", action="store_true", default=embedding.DEFAULT_QDRANT_ON_DISK_PAYLOAD, help="Store payload on disk.")
    parser.add_argument("--no-on-disk-payload", dest="on_disk_payload", action="store_false", help="Keep payload in RAM when possible.")
    parser.add_argument("--scalar-quantization", dest="enable_scalar_quantization", action="store_true", default=embedding.DEFAULT_QDRANT_ENABLE_SCALAR_QUANTIZATION, help="Enable int8 scalar quantization.")
    parser.add_argument("--no-scalar-quantization", dest="enable_scalar_quantization", action="store_false", help="Disable scalar quantization.")
    parser.add_argument("--scalar-quantization-always-ram", dest="scalar_quantization_always_ram", action="store_true", default=embedding.DEFAULT_QDRANT_SCALAR_QUANTIZATION_ALWAYS_RAM, help="Keep quantized vectors in RAM.")
    parser.add_argument("--no-scalar-quantization-always-ram", dest="scalar_quantization_always_ram", action="store_false", help="Allow quantized vectors to spill from RAM.")
    parser.add_argument("--qdrant-upsert-wait", dest="qdrant_upsert_wait", action="store_true", default=embedding.DEFAULT_QDRANT_UPSERT_WAIT, help="Wait for Qdrant upserts.")
    parser.add_argument("--no-qdrant-upsert-wait", dest="qdrant_upsert_wait", action="store_false", help="Do not wait for each Qdrant upsert.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    started_at = time.perf_counter()
    slicing_summary: Dict[str, Any] = {"status": "skipped"}
    embedding_summary: Dict[str, Any] = {"status": "skipped"}

    if not args.skip_slicing:
        print("[PIPELINE] Stage 1/2: clean and chunk txt files")
        slicing_summary = run_slicing_stage(args)
        print(
            f"[PIPELINE] Slicing done: processed={slicing_summary['processed']} "
            f"skipped={slicing_summary['skipped']} failed={slicing_summary['failed']}"
        )

    if not args.skip_embedding:
        print("[PIPELINE] Stage 2/2: vectorize chunks and sync Qdrant")
        embedding_summary = run_embedding_stage(args)
        print(
            f"[PIPELINE] Embedding done: processed={embedding_summary['processed']} "
            f"reused={embedding_summary['reused']} skipped={embedding_summary['skipped']} failed={embedding_summary['failed']}"
        )

    summary = {
        "slicing": slicing_summary,
        "embedding": embedding_summary,
        "total_seconds": time.perf_counter() - started_at,
    }
    summary_path = Path(args.chunk_output_dir) / "_pipeline_index.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PIPELINE] Summary written to: {summary_path}")
    print(f"[PIPELINE] Total time: {summary['total_seconds']:.1f}s")
    succeeded = slicing_summary.get("failed", 0) == 0 and embedding_summary.get("failed", 0) == 0
    if succeeded:
        print("[PIPELINE] DONE: slicing/vectorization/storage completed successfully.")
        if not args.skip_embedding:
            print(
                "[PIPELINE] Storage target: "
                f"collection={embedding_summary.get('collection', '')} "
                f"qdrant_url={embedding_summary.get('qdrant_url', '') or 'local'}"
            )
    else:
        print("[PIPELINE] FAILED: one or more slicing/vectorization/storage tasks failed.")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
