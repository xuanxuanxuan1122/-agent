"""Quick health check for the local Qdrant store and embedding index."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

DEFAULT_DB_PATH = r"D:\pychram\RAG2\rag_chunks_store\qdrant_local"
DEFAULT_URL = ""
DEFAULT_COLLECTION = "rag_local_chunks"
DEFAULT_INDEX_PATH = Path(r"D:\pychram\RAG2\rag_chunks_store\_embedding_index.json")
REQUIRED_PAYLOAD_FIELDS = [
    "chunk_uid",
    "chunk_level",
    "source_file",
    "doc_title",
    "retrieval_text",
]

def load_index_summary(index_path: Path) -> Dict[str, Any]:
    # The index file records whether each source file was processed or reused.
    if not index_path.exists():
        return {"exists": False}

    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"exists": True, "error": str(exc)}

    statuses = [item.get("status", "<missing>") for item in data if isinstance(item, dict)]
    qdrant_counts = [item.get("qdrant_point_count", 0) for item in data if isinstance(item, dict)]
    child_counts = [item.get("chunk_count", 0) for item in data if isinstance(item, dict)]
    parent_counts = [item.get("parent_chunk_count", 0) for item in data if isinstance(item, dict)]
    return {
        "exists": True,
        "record_count": len(data) if isinstance(data, list) else 0,
        "statuses": statuses,
        "qdrant_counts": qdrant_counts,
        "child_counts": child_counts,
        "parent_counts": parent_counts,
    }


def validate_payload_sample(client: Any, collection: str, sample_limit: int) -> int:
    if sample_limit <= 0:
        return 0
    points, _ = client.scroll(
        collection_name=collection,
        limit=sample_limit,
        with_payload=True,
        with_vectors=False,
    )
    missing_count = 0
    empty_text_count = 0
    for point in points:
        payload = point.payload or {}
        missing_fields = [field for field in REQUIRED_PAYLOAD_FIELDS if field not in payload]
        if missing_fields:
            missing_count += 1
            print(f"[WARN] Payload missing fields for point={point.id}: {missing_fields}")
        if not str(payload.get("retrieval_text", "") or payload.get("text", "")).strip():
            empty_text_count += 1
            print(f"[WARN] Empty retrieval text for point={point.id}")
    print(
        f"[INFO] Payload sample checked: sampled={len(points)}, "
        f"missing_payload_records={missing_count}, empty_text_records={empty_text_count}"
    )
    return missing_count + empty_text_count


def run_sparse_sanity_query(client: Any, collection: str, sanity_query: str, top_k: int) -> int:
    query = str(sanity_query or "").strip()
    if not query:
        return 0
    try:
        from qdrant_client import models as qmodels
        from ..ingest.embedding_qdrant import DEFAULT_SPARSE_VECTOR_NAME, build_sparse_vector
    except Exception as exc:
        print(f"[WARN] Sparse sanity query skipped because dependencies were unavailable: {exc}")
        return 0

    sparse_payload = build_sparse_vector(query)
    indices = [int(value) for value in sparse_payload.get("indices", [])]
    values = [float(value) for value in sparse_payload.get("values", [])]
    if not indices:
        print(f"[WARN] Sparse sanity query produced no terms: {query}")
        return 1
    try:
        response = client.query_points(
            collection_name=collection,
            query=qmodels.SparseVector(indices=indices, values=values),
            using=DEFAULT_SPARSE_VECTOR_NAME,
            limit=max(1, top_k),
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        print(f"[WARN] Sparse sanity query failed: {exc}")
        return 1

    hits = response.points
    print(f"[INFO] Sparse sanity query: {query}, hits={len(hits)}")
    for rank, point in enumerate(hits[:top_k], start=1):
        payload = point.payload or {}
        label = payload.get("doc_title") or payload.get("source_file") or point.id
        section = payload.get("section_title", "")
        print(f"[INFO]   #{rank} score={float(point.score):.4f} {label} {section}")
    return 0 if hits else 1


def main() -> int:
    # This script is intentionally lightweight: it checks files first, then the Qdrant collection.
    parser = argparse.ArgumentParser(description="Validate local Qdrant vector store status.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="Path to the local Qdrant storage directory, or ':memory:' for in-memory local mode.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Qdrant server URL, e.g. http://localhost:6333.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Path to embedding index JSON.")
    parser.add_argument("--sample-limit", type=int, default=20, help="Payload records to sample for completeness checks.")
    parser.add_argument("--sanity-query", default="人工智能 产业革命", help="Optional sparse sanity query used to check lexical recall.")
    parser.add_argument("--sanity-top-k", type=int, default=5, help="Top-k hits to show for the sparse sanity query.")
    args = parser.parse_args()

    db_path = args.db_path
    index_path = Path(args.index_path)

    if args.url:
        print(f"[INFO] Qdrant URL: {args.url}")
    elif args.db_path == ":memory:":
        print("[INFO] Qdrant local mode: :memory:")
    else:
        print(f"[INFO] Qdrant path: {db_path}")
    print(f"[INFO] Collection: {args.collection}")

    index_summary = load_index_summary(index_path)
    if not index_summary.get("exists"):
        print(f"[WARN] Index file not found: {index_path}")
    elif index_summary.get("error"):
        print(f"[WARN] Failed to parse index file: {index_summary['error']}")
    else:
        print(f"[INFO] Index file records: {index_summary['record_count']}")
        for status, qdrant_count, child_count, parent_count in zip(
            index_summary.get("statuses", []),
            index_summary.get("qdrant_counts", []),
            index_summary.get("child_counts", []),
            index_summary.get("parent_counts", []),
        ):
            print(
                f"[INFO] Index status={status}, "
                f"child_chunk_count={child_count}, "
                f"parent_chunk_count={parent_count}, "
                f"qdrant_point_count={qdrant_count}"
            )

    if not args.url and args.db_path != ":memory:" and not Path(db_path).exists():
        print("[FAIL] Local Qdrant storage directory does not exist yet.")
        print("[HINT] 这通常表示向量还没有真正写入本地 Qdrant。")
        return 1

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        print(f"[FAIL] Missing dependency qdrant-client: {exc}")
        print("[HINT] 请先安装 qdrant-client，否则无法校验本地 Qdrant。")
        return 1

    if args.url:
        client = QdrantClient(url=args.url)
    elif args.db_path == ":memory:":
        client = QdrantClient(":memory:")
    else:
        client = QdrantClient(path=str(db_path))
    try:
        exists = client.collection_exists(args.collection)
        print(f"[INFO] Collection exists: {exists}")
        if not exists:
            print("[FAIL] Qdrant storage exists, but the target collection was not found.")
            return 1

        collection_info = client.get_collection(args.collection)
        print(f"[INFO] Collection info: {collection_info}")

        count_result = client.count(collection_name=args.collection, exact=True)
        print(f"[INFO] Point count: {count_result.count}")
        validation_errors = 0
        validation_errors += validate_payload_sample(client, args.collection, args.sample_limit)
        validation_errors += run_sparse_sanity_query(client, args.collection, args.sanity_query, args.sanity_top_k)
    finally:
        client.close()

    if validation_errors:
        print(f"[WARN] Vector store validation finished with {validation_errors} warning-level issue(s).")
        return 2
    print("[OK] Vector store validation finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
