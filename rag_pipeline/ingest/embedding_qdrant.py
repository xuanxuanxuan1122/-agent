import argparse
import hashlib
import importlib.util
import gc
import json
import math
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CHUNK_INPUT = Path(r"D:\pychram\RAG2\rag_chunks_store")
DEFAULT_EMBED_OUTPUT = Path(os.getenv("RAG_EMBED_OUTPUT_DIR", str(DEFAULT_CHUNK_INPUT)))
DEFAULT_MODEL_PATH = os.getenv("QWEN3_EMBEDDING_MODEL_PATH", r"D:\Qwen3-Embedding-4B")
DEFAULT_DEVICE = os.getenv("QWEN3_EMBEDDING_DEVICE", "auto")
DEFAULT_DTYPE = os.getenv("QWEN3_EMBEDDING_DTYPE", "float16")
DEFAULT_ATTN_IMPL = os.getenv("QWEN3_EMBEDDING_ATTN_IMPL", "sdpa")
DEFAULT_MAX_LENGTH = int(os.getenv("QWEN3_EMBEDDING_MAX_LENGTH", "2048"))
DEFAULT_BATCH_SIZE = int(os.getenv("QWEN3_EMBEDDING_BATCH_SIZE", "24"))
DEFAULT_ENABLE_TF32 = os.getenv("QWEN3_EMBEDDING_TF32", "1") == "1"
DEFAULT_PIN_MEMORY = os.getenv("QWEN3_EMBEDDING_PIN_MEMORY", "1") == "1"
DEFAULT_PAD_TO_MULTIPLE_OF = max(0, int(os.getenv("QWEN3_EMBEDDING_PAD_TO_MULTIPLE_OF", "8")))
DEFAULT_REQUEST_INTERVAL = float(os.getenv("QWEN3_EMBEDDING_REQUEST_INTERVAL", "0"))
DEFAULT_MAX_RETRIES = int(os.getenv("QWEN3_EMBEDDING_MAX_RETRIES", "3"))
DEFAULT_INITIAL_DELAY = float(os.getenv("QWEN3_EMBEDDING_INITIAL_DELAY", "1"))
DEFAULT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "256"))
DEFAULT_QDRANT_UPSERT_WAIT = os.getenv("QDRANT_UPSERT_WAIT", "0") == "1"
DEFAULT_QDRANT_PATH = os.getenv("QDRANT_LOCAL_PATH", str(DEFAULT_CHUNK_INPUT / "qdrant_local"))
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
DEFAULT_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
DEFAULT_QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "0") == "1"
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "rag_local_chunks")
DEFAULT_QDRANT_DENSE_ON_DISK = os.getenv("QDRANT_DENSE_ON_DISK", "0") == "1"
DEFAULT_QDRANT_HNSW_ON_DISK = os.getenv("QDRANT_HNSW_ON_DISK", "0") == "1"
DEFAULT_QDRANT_HNSW_M = max(4, int(os.getenv("QDRANT_HNSW_M", "32")))
DEFAULT_QDRANT_HNSW_EF_CONSTRUCT = max(16, int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "256")))
DEFAULT_QDRANT_ON_DISK_PAYLOAD = os.getenv("QDRANT_ON_DISK_PAYLOAD", "0") == "1"
DEFAULT_QDRANT_ENABLE_SCALAR_QUANTIZATION = os.getenv("QDRANT_ENABLE_SCALAR_QUANTIZATION", "1") == "1"
DEFAULT_QDRANT_SCALAR_QUANTIZATION_ALWAYS_RAM = os.getenv("QDRANT_SCALAR_QUANTIZATION_ALWAYS_RAM", "1") == "1"
DEFAULT_MIN_QUALITY_SCORE = float(os.getenv("QDRANT_MIN_QUALITY_SCORE", "0.75"))
DEFAULT_MIN_PARENT_QUALITY_SCORE = float(os.getenv("QDRANT_MIN_PARENT_QUALITY_SCORE", "0.60"))
DEFAULT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
DEFAULT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "lexical")
DEFAULT_ENABLE_SPARSE_VECTORS = os.getenv("QDRANT_ENABLE_SPARSE_VECTORS", "1") == "1"
DEFAULT_STORE_ALL_CHUNKS = os.getenv("QDRANT_STORE_ALL_CHUNKS", "0") == "1"
DEFAULT_WRITE_EMBEDDED_JSON = os.getenv("WRITE_EMBEDDED_JSON", "1") == "1"
DEFAULT_STORE_TO_QDRANT = os.getenv("STORE_TO_QDRANT", "1") == "1"
DEFAULT_DROP_IF_EXISTS = os.getenv("QDRANT_DROP_IF_EXISTS", "0") == "1"
DEFAULT_FORCE_REPROCESS = os.getenv("FORCE_REPROCESS", "0") == "1"
DEFAULT_REUPSERT_EXISTING = os.getenv("QDRANT_REUPSERT_EXISTING", "0") == "1"
DEFAULT_ENABLE_METADATA_ENRICH = os.getenv("RAG_ENABLE_METADATA_ENRICH", "1") == "1"
DEFAULT_SPARSE_TOKENIZER = (os.getenv("RAG_SPARSE_TOKENIZER", "auto") or "auto").strip().lower()
DEFAULT_SPARSE_KEEP_COMPAT_NGRAMS = os.getenv("RAG_SPARSE_KEEP_COMPAT_NGRAMS", "1") == "1"
DEFAULT_BGE_M3_MODEL_PATH = os.getenv(
    "BGE_M3_MODEL_PATH",
    r"D:\BGE-M3" if Path(r"D:\BGE-M3").exists() else "",
).strip()
DEFAULT_BGE_M3_DEVICE = os.getenv("BGE_M3_DEVICE", "auto").strip() or "auto"
DEFAULT_BGE_M3_BATCH_SIZE = max(1, int(os.getenv("BGE_M3_BATCH_SIZE", "16")))
DEFAULT_BGE_M3_QUERY_MAX_LENGTH = max(32, int(os.getenv("BGE_M3_QUERY_MAX_LENGTH", "512")))
DEFAULT_BGE_M3_PASSAGE_MAX_LENGTH = max(32, int(os.getenv("BGE_M3_PASSAGE_MAX_LENGTH", "512")))
DEFAULT_BGE_M3_USE_FP16 = os.getenv("BGE_M3_USE_FP16", "1") == "1"
DEFAULT_BGE_DENSE_VECTOR_NAME = os.getenv("QDRANT_BGE_DENSE_VECTOR_NAME", "bge_dense").strip() or "bge_dense"
DEFAULT_BGE_SPARSE_VECTOR_NAME = os.getenv("QDRANT_BGE_SPARSE_VECTOR_NAME", "bge_sparse").strip() or "bge_sparse"
DEFAULT_EMBED_SCOPE = (os.getenv("RAG_EMBED_SCOPE", "qdrant") or "qdrant").strip().lower()
DEFAULT_PREVIEW_TOP_K = max(0, int(os.getenv("RAG_EMBED_PREVIEW_TOP_K", "0")))
if DEFAULT_EMBED_SCOPE not in {"qdrant", "all"}:
    DEFAULT_EMBED_SCOPE = "qdrant"

HF_CACHE_DIR = Path(os.getenv("RAG_HF_CACHE_DIR", str(DEFAULT_CHUNK_INPUT / ".hf_cache")))
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SPLIT_RE = re.compile(r"[，。！？!?、；;:：()（）【】\[\]{}<>\s,/\\|]+")
_STOP_RE = re.compile(r"(?:可以|能够|如何|怎么|怎样|哪些|是否|需要|确保|以及|和|与|的|了|在|对|通过|从而|因此|如果|因为|所以|而且|并且|并|会|就|都|更|较|很|为|把|被|是)")
_JIEBA_MODULE = None


def _require_torch():
    try:
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'torch'. Install a CUDA-enabled PyTorch build before running local embeddings."
        ) from exc
    return torch, torch_functional


def _require_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'transformers'. Install it before running local embeddings.") from exc
    return AutoModel, AutoTokenizer


def _require_qdrant_client():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client import models as qmodels
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'qdrant-client'. Install it before writing vectors to Qdrant.") from exc
    return QdrantClient, qmodels


def _require_flagembedding():
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'FlagEmbedding'. Install it before running BGE-M3 retrieval.") from exc
    return BGEM3FlagModel


def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "config.json").exists() and (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer_config.json").exists()
        or (path / "vocab.json").exists()
    )


def resolve_local_model_dir(model_name_or_path: str | Path) -> Path:
    model_path = Path(model_name_or_path)
    if _looks_like_model_dir(model_path):
        return model_path
    if not model_path.exists() or not model_path.is_dir():
        return model_path
    candidates: List[Path] = []
    root_depth = len(model_path.parts)
    for config_path in model_path.rglob("config.json"):
        candidate = config_path.parent
        if len(candidate.parts) - root_depth > 4:
            continue
        if _looks_like_model_dir(candidate):
            candidates.append(candidate)
    if not candidates:
        return model_path
    candidates.sort(key=lambda item: (len(item.parts), str(item).lower()))
    return candidates[0]


def _is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def collect_runtime_dependency_issues(
    model_name_or_path: str | Path | None = None,
    device: str = "auto",
    require_qdrant: bool = False,
    bge_m3_model_path: str | Path | None = None,
    require_bge_m3: bool = False,
) -> List[str]:
    issues: List[str] = []
    if not _is_module_available("torch"):
        issues.append("缺少依赖 torch，请先安装可用的 PyTorch。")
    if not _is_module_available("transformers"):
        issues.append("缺少依赖 transformers，请先安装 transformers。")
    if require_qdrant and not _is_module_available("qdrant_client"):
        issues.append("缺少依赖 qdrant-client，请先安装 qdrant-client。")

    if require_bge_m3 and not _is_module_available("FlagEmbedding"):
        issues.append("Missing dependency 'FlagEmbedding'. Install it before running BGE-M3.")

    if model_name_or_path:
        raw_model_path = Path(model_name_or_path)
        resolved_model_path = resolve_local_model_dir(raw_model_path)
        if not raw_model_path.exists():
            issues.append(f"本地 embedding 模型目录不存在: {raw_model_path}")
        elif not _looks_like_model_dir(resolved_model_path):
            issues.append(f"本地 embedding 模型目录存在，但没有找到可直接加载的模型文件 (当前路径: {raw_model_path})。")

    if require_bge_m3:
        if not bge_m3_model_path:
            issues.append("BGE-M3 is enabled, but no local model path was provided.")
        else:
            raw_bge_path = Path(bge_m3_model_path)
            resolved_bge_path = resolve_local_model_dir(raw_bge_path)
            if not raw_bge_path.exists():
                issues.append(f"BGE-M3 local model directory does not exist: {raw_bge_path}")
            elif not _looks_like_model_dir(resolved_bge_path):
                issues.append(
                    f"BGE-M3 model directory exists, but no directly loadable model files were found: {raw_bge_path}"
                )

    if device not in {"", "auto", "cpu"} and device.startswith("cuda") and _is_module_available("torch"):
        try:
            torch_module, _ = _require_torch()
            if not torch_module.cuda.is_available():
                issues.append(f"请求使用 CUDA 设备 '{device}'，但当前环境没有可用的 CUDA。")
        except RuntimeError:
            pass
    return issues


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_chunk_uid(source_file: str, chunk: Dict[str, Any]) -> str:
    seed = "|".join(
        [
            str(source_file or ""),
            str(chunk.get("doc_id", "")),
            str(chunk.get("section_id", "")),
            str(chunk.get("chunk_in_section", "")),
            str(chunk.get("embedding_text", chunk.get("text", "")) or ""),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def build_parent_chunk_uid(source_file: str, chunk: Dict[str, Any]) -> str:
    child_uids = [str(uid).strip() for uid in chunk.get("child_chunk_uids", []) if str(uid).strip()]
    seed = "|".join(
        [
            str(source_file or ""),
            str(chunk.get("doc_id", "")),
            str(chunk.get("section_id", "")),
            str(chunk.get("parent_chunk_index", chunk.get("chunk_in_section", ""))),
            str(chunk.get("chunk_level", "parent")),
            ",".join(child_uids),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def resolve_chunk_uid(source_file: str, chunk: Dict[str, Any]) -> str:
    chunk_uid = str(chunk.get("chunk_uid", "") or "").strip()
    if chunk_uid:
        return chunk_uid
    if str(chunk.get("chunk_level", "child")) == "parent":
        return build_parent_chunk_uid(source_file, chunk)
    return build_chunk_uid(source_file, chunk)


def ensure_chunk_uids(chunks: List[Dict[str, Any]], source_file: str) -> List[Dict[str, Any]]:
    for chunk in chunks:
        chunk["chunk_uid"] = resolve_chunk_uid(source_file, chunk)
    return chunks


def extract_alias_terms(chunk: Dict[str, Any]) -> List[str]:
    values = []
    for key in ["doc_title", "section_title", "semantic_role", "table_family", "source"]:
        val = normalize_text(chunk.get(key, ""))
        if val:
            values.append(val)
    header_path = chunk.get("header_path", [])
    if isinstance(header_path, list):
        values.extend(normalize_text(x) for x in header_path if normalize_text(x))
    seen = set()
    deduped = []
    for val in values:
        if val not in seen:
            seen.add(val)
            deduped.append(val)
    return deduped[:20]


def extract_keywords(text: str, max_keywords: int = 20) -> List[str]:
    text = normalize_text(text).lower()
    if not text:
        return []
    if _CJK_RE.search(text):
        text = _STOP_RE.sub(" ", text)
    terms = []
    for part in _SPLIT_RE.split(text):
        part = part.strip()
        if len(part) < 2:
            continue
        terms.append(part)
    seen = set()
    deduped = []
    for term in sorted(terms, key=len, reverse=True):
        if term not in seen:
            seen.add(term)
            deduped.append(term)
        if len(deduped) >= max_keywords:
            break
    return deduped


def build_sparse_source_text(chunk: Dict[str, Any]) -> str:
    parts = [
        normalize_text(chunk.get("doc_title", "")),
        normalize_text(chunk.get("section_title", "")),
        normalize_text(chunk.get("header_path_text", "")),
        normalize_text(chunk.get("retrieval_text", "") or chunk.get("embedding_text", "") or chunk.get("text", "")),
        " ".join(str(value).strip() for value in chunk.get("alias_terms", []) if str(value).strip()),
        " ".join(str(value).strip() for value in chunk.get("keywords", []) if str(value).strip()),
    ]
    return "\n".join(part for part in parts if part)


def _load_jieba_module():
    global _JIEBA_MODULE
    if _JIEBA_MODULE is not None:
        return _JIEBA_MODULE
    if DEFAULT_SPARSE_TOKENIZER not in {"auto", "jieba"}:
        return None
    if importlib.util.find_spec("jieba") is None:
        return None
    import jieba  # type: ignore

    _JIEBA_MODULE = jieba
    return _JIEBA_MODULE


def iter_sparse_term_weights(text: str) -> List[Tuple[str, float]]:
    normalized = normalize_text(text).lower()
    if not normalized:
        return []

    terms: List[Tuple[str, float]] = []
    jieba_module = _load_jieba_module()
    for part in _SPLIT_RE.split(normalized):
        part = part.strip()
        if len(part) < 2:
            continue
        if _CJK_RE.search(part):
            subparts = [seg.strip() for seg in re.split(r"[与和及、/]", part) if len(seg.strip()) >= 2]
            if not subparts:
                subparts = [part]
            for seg in subparts:
                terms.append((seg, 0.70))
                if jieba_module is not None:
                    for token in jieba_module.lcut(seg, HMM=False):
                        token = token.strip()
                        if len(token) >= 2:
                            terms.append((token, 1.35))
                if DEFAULT_SPARSE_KEEP_COMPAT_NGRAMS:
                    for n in (2, 3):
                        if len(seg) >= n:
                            for i in range(len(seg) - n + 1):
                                grams = seg[i : i + n]
                                # Keep old collection compatibility, but avoid letting noisy grams dominate.
                                terms.append((grams, 0.45 if n == 2 else 0.65))
        else:
            terms.append((part, 1.0))
    return terms


def iter_sparse_terms(text: str) -> List[str]:
    return [term for term, _ in iter_sparse_term_weights(text)]


def stable_sparse_term_id(term: str) -> int:
    digest = hashlib.sha1(term.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % 2147483647


def build_sparse_vector(text: str) -> Dict[str, List[float] | List[int]]:
    term_weights: Dict[int, float] = {}
    for term, weight in iter_sparse_term_weights(text):
        term_id = stable_sparse_term_id(term)
        term_weights[term_id] = term_weights.get(term_id, 0.0) + max(0.0, float(weight))

    if not term_weights:
        return {"indices": [], "values": []}

    indices = sorted(term_weights.keys())
    values = [1.0 + math.log1p(term_weights[index]) for index in indices]
    return {"indices": indices, "values": values}


def build_retrieval_text(chunk: Dict[str, Any]) -> str:
    title = normalize_text(chunk.get("doc_title", ""))
    section = normalize_text(chunk.get("section_title", ""))
    header_path = chunk.get("header_path", [])
    header_text = " > ".join(str(x).strip() for x in header_path if str(x).strip()) if isinstance(header_path, list) else normalize_text(header_path)
    body = normalize_text(chunk.get("embedding_text") or chunk.get("text") or "")
    aliases = " | ".join(extract_alias_terms(chunk))
    parts = [p for p in [title, header_text, section, aliases, body] if p]
    return "\n".join(parts)


def enrich_chunk_metadata(chunk: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    enriched = dict(chunk)
    retrieval_text = build_retrieval_text(enriched)
    keywords = extract_keywords(retrieval_text)
    aliases = extract_alias_terms(enriched)
    header_path = enriched.get("header_path", [])
    header_path_text = " > ".join(str(v).strip() for v in header_path if str(v).strip()) if isinstance(header_path, list) else normalize_text(header_path)

    enriched["chunk_level"] = str(enriched.get("chunk_level") or "child")
    enriched["chunk_uid"] = resolve_chunk_uid(source_file, enriched)
    enriched["parent_chunk_uid"] = str(enriched.get("parent_chunk_uid", "") or "")
    enriched["parent_chunk_index"] = int(enriched.get("parent_chunk_index", 0) or 0)
    enriched["parent_chunk_count"] = int(enriched.get("parent_chunk_count", 0) or 0)
    enriched["child_chunk_uids"] = [str(value).strip() for value in enriched.get("child_chunk_uids", []) if str(value).strip()]
    enriched["child_chunk_count"] = int(enriched.get("child_chunk_count", len(enriched["child_chunk_uids"])) or 0)
    enriched["retrieval_text"] = retrieval_text
    enriched["embedding_text"] = normalize_text(enriched.get("embedding_text") or retrieval_text)
    enriched["header_path_text"] = header_path_text
    enriched["alias_terms"] = aliases
    enriched["keywords"] = keywords
    enriched["doc_type"] = normalize_text(enriched.get("doc_type") or enriched.get("semantic_role") or enriched.get("chunk_type") or "")
    enriched["source_file_name"] = Path(source_file).name
    enriched["sparse_text"] = build_sparse_source_text(enriched)
    enriched["sparse_vector"] = build_sparse_vector(enriched["sparse_text"])
    enriched["metadata_version"] = "stage2"
    return enriched


def should_skip_qdrant_chunk(chunk: Dict[str, Any]) -> bool:
    quality_flags = set(chunk.get("quality_flags", []))
    quality_score = float(chunk.get("quality_score", 1.0))
    embedding_text = (chunk.get("embedding_text") or chunk.get("text") or "").strip()
    chunk_level = str(chunk.get("chunk_level", "child") or "child")
    min_quality_score = DEFAULT_MIN_PARENT_QUALITY_SCORE if chunk_level == "parent" else DEFAULT_MIN_QUALITY_SCORE
    noise_ratio = float(chunk.get("ocr_noise_ratio", 0.0) or 0.0)
    info_density = float(chunk.get("info_density", 0.0) or 0.0)
    noise_score = float(chunk.get("noise_score", 0.0) or 0.0)

    if not embedding_text:
        return True
    if noise_ratio >= 0.85:
        return True
    if noise_score >= 0.90 and info_density < 0.28:
        return True
    severe_flags = {
        "empty_chunk",
        "empty_table",
        "bad_table",
        "table_glued",
        "table_misaligned",
        "table_header_glued",
        "table_sparse",
        "table_overflow",
        "table_low_confidence",
        "ocr_noise",
        "garbled_text",
    }
    if severe_flags & quality_flags:
        return True
    if {"visual_noise_heavy", "low_info_density", "promotional_page"} & quality_flags and info_density < 0.30:
        return True
    if not DEFAULT_STORE_ALL_CHUNKS:
        if not chunk.get("is_retrieval_eligible", True):
            return True
        if quality_score < min_quality_score:
            return True
    return False


def filter_chunks_for_qdrant(chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    kept, skipped = [], 0
    for chunk in chunks:
        if should_skip_qdrant_chunk(chunk):
            skipped += 1
            continue
        kept.append(chunk)
    return kept, skipped


class QwenLocalEmbeddingModel:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        dtype: str = "float16",
        attn_implementation: str = "sdpa",
        max_length: int = 2048,
        normalize: bool = True,
        enable_tf32: bool = DEFAULT_ENABLE_TF32,
        pin_memory: bool = DEFAULT_PIN_MEMORY,
        pad_to_multiple_of: int = DEFAULT_PAD_TO_MULTIPLE_OF,
    ):
        self.model_name_or_path = model_name_or_path
        self.device_preference = device
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.max_length = max_length
        self.normalize = normalize
        self.enable_tf32 = enable_tf32
        self.pin_memory = pin_memory
        self.pad_to_multiple_of = max(0, int(pad_to_multiple_of or 0))
        self._tokenizer = None
        self._model = None
        self._device = None
        self._resolved_model_path = None

    def _resolve_device(self, torch_module):
        if self.device_preference not in {"", "auto"}:
            if self.device_preference.startswith("cuda") and not torch_module.cuda.is_available():
                raise RuntimeError("CUDA device was requested, but no CUDA device is available.")
            return self.device_preference
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"

    def _resolve_dtype(self, torch_module, device: str):
        if self.dtype_name == "bfloat16" and hasattr(torch_module, "bfloat16"):
            return torch_module.bfloat16
        if self.dtype_name == "float32":
            return torch_module.float32
        if device.startswith("cuda"):
            return torch_module.float16
        return torch_module.float32

    def _configure_torch_backends(self, torch_module) -> None:
        if not self._device or not str(self._device).startswith("cuda"):
            return
        if self.enable_tf32:
            try:
                torch_module.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                torch_module.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            if hasattr(torch_module, "set_float32_matmul_precision"):
                try:
                    torch_module.set_float32_matmul_precision("high")
                except Exception:
                    pass
        try:
            torch_module.backends.cudnn.benchmark = True
        except Exception:
            pass

    def _move_encoded_to_device(self, encoded: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        use_cuda = bool(self._device and str(self._device).startswith("cuda"))
        for key, value in encoded.items():
            tensor = value
            if use_cuda and self.pin_memory and hasattr(tensor, "pin_memory"):
                try:
                    tensor = tensor.pin_memory()
                except Exception:
                    tensor = value
            moved[key] = tensor.to(self._device, non_blocking=use_cuda)
        return moved

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return
        torch_module, _ = _require_torch()
        AutoModel, AutoTokenizer = _require_transformers()
        self._device = self._resolve_device(torch_module)
        resolved_dtype = self._resolve_dtype(torch_module, self._device)
        self._configure_torch_backends(torch_module)
        resolved_model_path = resolve_local_model_dir(self.model_name_or_path)
        self._resolved_model_path = str(resolved_model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._resolved_model_path,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model_kwargs = {"trust_remote_code": True, "local_files_only": True, "dtype": resolved_dtype}
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        try:
            self._model = AutoModel.from_pretrained(self._resolved_model_path, **model_kwargs)
        except TypeError:
            legacy_kwargs = dict(model_kwargs)
            legacy_kwargs.pop("dtype", None)
            legacy_kwargs["torch_dtype"] = resolved_dtype
            try:
                self._model = AutoModel.from_pretrained(self._resolved_model_path, **legacy_kwargs)
            except TypeError:
                legacy_kwargs.pop("attn_implementation", None)
                self._model = AutoModel.from_pretrained(self._resolved_model_path, **legacy_kwargs)
        self._model.to(self._device)
        self._model.eval()

    def _last_token_pool(self, last_hidden_state, attention_mask, torch_module):
        left_padded = bool((attention_mask[:, -1] == 1).all().item())
        if left_padded:
            return last_hidden_state[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indexes = torch_module.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        return last_hidden_state[batch_indexes, sequence_lengths]

    def encode(self, texts: List[str], batch_size: int = 6) -> List[List[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        torch_module, torch_functional = _require_torch()
        effective_batch_size = max(1, batch_size)
        sanitized_texts = [str(text or "").strip() for text in texts]
        ordered_items = sorted(enumerate(sanitized_texts), key=lambda item: (len(item[1]), item[0]))
        results: List[List[float] | None] = [None] * len(texts)
        start = 0
        while start < len(ordered_items):
            stop = min(start + effective_batch_size, len(ordered_items))
            batch_items = ordered_items[start:stop]
            batch_texts = [text for _, text in batch_items]
            try:
                with torch_module.inference_mode():
                    encoded = self._tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        pad_to_multiple_of=self.pad_to_multiple_of if self.pad_to_multiple_of > 0 and str(self._device).startswith("cuda") else None,
                        return_tensors="pt",
                    )
                    encoded = self._move_encoded_to_device(encoded)
                    outputs = self._model(**encoded)
                    pooled = self._last_token_pool(outputs.last_hidden_state, encoded["attention_mask"], torch_module)
                    if self.normalize:
                        pooled = torch_functional.normalize(pooled, p=2, dim=1)
                    batch_embeddings = pooled.float().cpu().tolist()
                for (original_idx, _), embedding in zip(batch_items, batch_embeddings):
                    results[original_idx] = embedding
                start = stop
            except RuntimeError as exc:
                error_text = str(exc).lower()
                if "out of memory" in error_text and effective_batch_size > 1:
                    effective_batch_size = max(1, effective_batch_size // 2)
                    if self._device and str(self._device).startswith("cuda"):
                        torch_module.cuda.empty_cache()
                    continue
                raise RuntimeError(f"Qwen local embedding failed: {exc}") from exc
        if any(embedding is None for embedding in results):
            raise RuntimeError("Embedding generation returned incomplete results.")
        return results

    def close(self) -> None:
        torch_module = None
        if self._model is not None:
            self._model = None
        if self._tokenizer is not None:
            self._tokenizer = None
        self._device = None
        self._resolved_model_path = None
        try:
            torch_module, _ = _require_torch()
        except Exception:
            torch_module = None
        if torch_module is not None and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        gc.collect()


class BgeM3EmbeddingModel:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = DEFAULT_BGE_M3_DEVICE,
        batch_size: int = DEFAULT_BGE_M3_BATCH_SIZE,
        query_max_length: int = DEFAULT_BGE_M3_QUERY_MAX_LENGTH,
        passage_max_length: int = DEFAULT_BGE_M3_PASSAGE_MAX_LENGTH,
        use_fp16: bool = DEFAULT_BGE_M3_USE_FP16,
        normalize: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.device_preference = device
        self.batch_size = max(1, int(batch_size or DEFAULT_BGE_M3_BATCH_SIZE))
        self.query_max_length = max(32, int(query_max_length or DEFAULT_BGE_M3_QUERY_MAX_LENGTH))
        self.passage_max_length = max(32, int(passage_max_length or DEFAULT_BGE_M3_PASSAGE_MAX_LENGTH))
        self.use_fp16 = bool(use_fp16)
        self.normalize = bool(normalize)
        self._model = None
        self._device = None
        self._resolved_model_path = None

    def _resolve_device(self) -> str:
        torch_module, _ = _require_torch()
        if self.device_preference not in {"", "auto"}:
            if self.device_preference.startswith("cuda") and not torch_module.cuda.is_available():
                raise RuntimeError("CUDA device was requested for BGE-M3, but no CUDA device is available.")
            return self.device_preference
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"

    def _ensure_loaded(self):
        if self._model is not None:
            return
        BGEM3FlagModel = _require_flagembedding()
        self._device = self._resolve_device()
        resolved_model_path = resolve_local_model_dir(self.model_name_or_path)
        self._resolved_model_path = str(resolved_model_path)
        self._model = BGEM3FlagModel(
            self._resolved_model_path,
            normalize_embeddings=self.normalize,
            use_fp16=bool(self.use_fp16 and str(self._device).startswith("cuda")),
            devices=self._device,
            batch_size=self.batch_size,
            query_max_length=self.query_max_length,
            passage_max_length=self.passage_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            cache_dir=str(HF_CACHE_DIR),
        )

    def _normalize_sparse_payloads(self, lexical_weights: Any) -> List[Dict[str, List[float] | List[int]]]:
        if lexical_weights is None:
            return []
        items = lexical_weights if isinstance(lexical_weights, list) else [lexical_weights]
        normalized_payloads: List[Dict[str, List[float] | List[int]]] = []
        for item in items:
            term_weights: Dict[int, float] = {}
            if isinstance(item, dict):
                for token_id, weight in item.items():
                    try:
                        sparse_id = int(token_id)
                    except Exception:
                        sparse_id = stable_sparse_term_id(str(token_id))
                    term_weights[sparse_id] = max(term_weights.get(sparse_id, 0.0), float(weight))
            indices = sorted(term_weights.keys())
            values = [float(term_weights[index]) for index in indices]
            normalized_payloads.append({"indices": indices, "values": values})
        return normalized_payloads

    def encode_corpus(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        return_dense: bool = True,
        return_sparse: bool = True,
    ) -> Dict[str, Any]:
        if not texts:
            return {"dense_vecs": [], "sparse_vectors": []}
        self._ensure_loaded()
        outputs = self._model.encode_corpus(
            list(texts),
            batch_size=max(1, int(batch_size or self.batch_size)),
            max_length=self.passage_max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )
        dense_vecs = outputs.get("dense_vecs")
        lexical_weights = outputs.get("lexical_weights")
        return {
            "dense_vecs": dense_vecs.tolist() if dense_vecs is not None else [],
            "sparse_vectors": self._normalize_sparse_payloads(lexical_weights) if return_sparse else [],
        }

    def encode_queries(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        return_dense: bool = True,
        return_sparse: bool = True,
    ) -> Dict[str, Any]:
        if not texts:
            return {"dense_vecs": [], "sparse_vectors": []}
        self._ensure_loaded()
        outputs = self._model.encode_queries(
            list(texts),
            batch_size=max(1, int(batch_size or self.batch_size)),
            max_length=self.query_max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )
        dense_vecs = outputs.get("dense_vecs")
        lexical_weights = outputs.get("lexical_weights")
        return {
            "dense_vecs": dense_vecs.tolist() if dense_vecs is not None else [],
            "sparse_vectors": self._normalize_sparse_payloads(lexical_weights) if return_sparse else [],
        }

    def close(self) -> None:
        if self._model is not None and hasattr(self._model, "stop_self_pool"):
            try:
                self._model.stop_self_pool()
            except Exception:
                pass
        self._model = None
        self._device = None
        self._resolved_model_path = None
        try:
            torch_module, _ = _require_torch()
        except Exception:
            torch_module = None
        if torch_module is not None and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        gc.collect()


class BgeM3IngestHelper:
    def __init__(
        self,
        model_path: str = DEFAULT_BGE_M3_MODEL_PATH,
        device: str = DEFAULT_BGE_M3_DEVICE,
        batch_size: int = DEFAULT_BGE_M3_BATCH_SIZE,
        query_max_length: int = DEFAULT_BGE_M3_QUERY_MAX_LENGTH,
        passage_max_length: int = DEFAULT_BGE_M3_PASSAGE_MAX_LENGTH,
        use_fp16: bool = DEFAULT_BGE_M3_USE_FP16,
    ):
        self.model_path = str(model_path or "").strip()
        self.enabled = bool(self.model_path)
        self.device = device
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.use_fp16 = use_fp16
        self._encoder: Optional[BgeM3EmbeddingModel] = None

    def _ensure_encoder(self) -> Optional[BgeM3EmbeddingModel]:
        if not self.enabled:
            return None
        if self._encoder is None:
            self._encoder = BgeM3EmbeddingModel(
                model_name_or_path=self.model_path,
                device=self.device,
                batch_size=self.batch_size,
                query_max_length=self.query_max_length,
                passage_max_length=self.passage_max_length,
                use_fp16=self.use_fp16,
            )
        return self._encoder

    def enrich_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return chunk
        enriched = dict(chunk)
        enriched["bge_m3_ready"] = bool(
            isinstance(enriched.get("bge_dense_embedding"), list)
            and enriched.get("bge_dense_embedding")
            and isinstance(enriched.get("bge_sparse_vector"), dict)
            and enriched["bge_sparse_vector"].get("indices")
        )
        return enriched

    def ensure_artifacts_for_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_retries: int,
        initial_delay: float,
        request_interval: float,
    ) -> List[Dict[str, Any]]:
        encoder = self._ensure_encoder()
        if encoder is None or not chunks:
            return chunks

        missing_indexes: List[int] = []
        pending_texts: List[str] = []
        seen_texts: set[str] = set()
        cache: Dict[str, Dict[str, Any]] = {}
        for idx, chunk in enumerate(chunks):
            has_dense = isinstance(chunk.get("bge_dense_embedding"), list) and bool(chunk.get("bge_dense_embedding"))
            sparse_payload = chunk.get("bge_sparse_vector") or {}
            has_sparse = isinstance(sparse_payload, dict) and bool(sparse_payload.get("indices"))
            if has_dense and has_sparse:
                chunk["bge_m3_ready"] = True
                continue
            text = (chunk.get("embedding_text") or chunk.get("text") or "").strip()
            if not text:
                continue
            missing_indexes.append(idx)
            if text not in seen_texts:
                pending_texts.append(text)
                seen_texts.add(text)

        if not missing_indexes:
            return chunks

        delay = initial_delay
        for start in range(0, len(pending_texts), encoder.batch_size):
            batch = pending_texts[start:start + encoder.batch_size]
            for attempt in range(1, max_retries + 1):
                try:
                    batch_outputs = encoder.encode_corpus(batch, batch_size=encoder.batch_size, return_dense=True, return_sparse=True)
                    dense_vecs = list(batch_outputs.get("dense_vecs", []) or [])
                    sparse_vectors = list(batch_outputs.get("sparse_vectors", []) or [])
                    if len(dense_vecs) != len(batch) or len(sparse_vectors) != len(batch):
                        raise RuntimeError(
                            f"BGE-M3 batch size mismatch: expected {len(batch)}, got dense={len(dense_vecs)} sparse={len(sparse_vectors)}"
                        )
                    for text, dense_vec, sparse_vector in zip(batch, dense_vecs, sparse_vectors):
                        cache[text] = {
                            "bge_dense_embedding": dense_vec,
                            "bge_sparse_vector": sparse_vector,
                        }
                    break
                except Exception as exc:
                    if attempt >= max_retries:
                        raise RuntimeError(f"BGE-M3 corpus encoding failed: {exc}") from exc
                    print(f"[WARN] BGE-M3 batch failed, retry {attempt}/{max_retries} after {delay:.1f}s: {exc}")
                    sleep_with_jitter(delay)
                    delay *= 2
            if request_interval > 0 and start + encoder.batch_size < len(pending_texts):
                time.sleep(request_interval)

        for idx in missing_indexes:
            text = (chunks[idx].get("embedding_text") or chunks[idx].get("text") or "").strip()
            if text not in cache:
                continue
            chunks[idx]["bge_dense_embedding"] = cache[text]["bge_dense_embedding"]
            chunks[idx]["bge_sparse_vector"] = cache[text]["bge_sparse_vector"]
            chunks[idx]["bge_m3_ready"] = True
        return chunks

    def encode_queries(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        return_dense: bool = True,
        return_sparse: bool = True,
    ) -> Dict[str, Any]:
        encoder = self._ensure_encoder()
        if encoder is None:
            return {"dense_vecs": [], "sparse_vectors": []}
        return encoder.encode_queries(
            texts,
            batch_size=batch_size or encoder.batch_size,
            return_dense=return_dense,
            return_sparse=return_sparse,
        )

    def close(self) -> None:
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None


class QdrantLocalVectorStore:
    def __init__(
        self,
        db_path: str | Path,
        collection_name: str,
        drop_if_exists: bool = False,
        url: str = "",
        api_key: str = "",
        prefer_grpc: bool = False,
        dense_vector_name: str = DEFAULT_DENSE_VECTOR_NAME,
        sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME,
        enable_sparse_vectors: bool = DEFAULT_ENABLE_SPARSE_VECTORS,
        bge_dense_vector_name: str = DEFAULT_BGE_DENSE_VECTOR_NAME,
        bge_sparse_vector_name: str = DEFAULT_BGE_SPARSE_VECTOR_NAME,
        enable_bge_dense: bool = False,
        enable_bge_sparse: bool = False,
        dense_on_disk: bool = DEFAULT_QDRANT_DENSE_ON_DISK,
        hnsw_on_disk: bool = DEFAULT_QDRANT_HNSW_ON_DISK,
        hnsw_m: int = DEFAULT_QDRANT_HNSW_M,
        hnsw_ef_construct: int = DEFAULT_QDRANT_HNSW_EF_CONSTRUCT,
        on_disk_payload: bool = DEFAULT_QDRANT_ON_DISK_PAYLOAD,
        enable_scalar_quantization: bool = DEFAULT_QDRANT_ENABLE_SCALAR_QUANTIZATION,
        scalar_quantization_always_ram: bool = DEFAULT_QDRANT_SCALAR_QUANTIZATION_ALWAYS_RAM,
        upsert_wait: bool = DEFAULT_QDRANT_UPSERT_WAIT,
    ):
        self.db_path = str(db_path)
        self.collection_name = collection_name
        self.drop_if_exists = drop_if_exists
        self.url = str(url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.prefer_grpc = prefer_grpc
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.enable_sparse_vectors = enable_sparse_vectors
        self.bge_dense_vector_name = bge_dense_vector_name
        self.bge_sparse_vector_name = bge_sparse_vector_name
        self.enable_bge_dense = enable_bge_dense
        self.enable_bge_sparse = enable_bge_sparse
        self.dense_on_disk = dense_on_disk
        self.hnsw_on_disk = hnsw_on_disk
        self.hnsw_m = max(4, int(hnsw_m or DEFAULT_QDRANT_HNSW_M))
        self.hnsw_ef_construct = max(16, int(hnsw_ef_construct or DEFAULT_QDRANT_HNSW_EF_CONSTRUCT))
        self.on_disk_payload = on_disk_payload
        self.enable_scalar_quantization = enable_scalar_quantization
        self.scalar_quantization_always_ram = scalar_quantization_always_ram
        self.upsert_wait = upsert_wait
        self._client = None
        self._collection_ready = False
        self._vector_dims: Dict[str, int] = {}

    def _build_quantization_config(self, qmodels: Any):
        if not self.enable_scalar_quantization:
            return None
        return qmodels.ScalarQuantization(
            scalar=qmodels.ScalarQuantizationConfig(
                type=qmodels.ScalarType.INT8,
                always_ram=self.scalar_quantization_always_ram,
            )
        )

    def _build_dense_vector_params(self, vector_dim: int, qmodels: Any):
        return qmodels.VectorParams(
            size=vector_dim,
            distance=qmodels.Distance.COSINE,
            on_disk=self.dense_on_disk,
            quantization_config=self._build_quantization_config(qmodels),
        )

    def _build_dense_vector_diff(self, qmodels: Any):
        return qmodels.VectorParamsDiff(
            on_disk=self.dense_on_disk,
            quantization_config=self._build_quantization_config(qmodels),
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        QdrantClient, _ = _require_qdrant_client()
        if self.url:
            client_kwargs = {"url": self.url, "prefer_grpc": self.prefer_grpc, "check_compatibility": False}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            self._client = QdrantClient(**client_kwargs)
        elif self.db_path == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=self.db_path)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _extract_existing_dense_vector_names(self, collection_info: Any) -> set[str]:
        vectors = getattr(getattr(collection_info, "config", None), "params", None)
        vectors = getattr(vectors, "vectors", None)
        if isinstance(vectors, dict):
            return {str(name) for name in vectors.keys()}
        if vectors is not None:
            return {self.dense_vector_name}
        return set()

    def _extract_existing_sparse_vector_names(self, collection_info: Any) -> set[str]:
        sparse_vectors = getattr(getattr(collection_info, "config", None), "params", None)
        sparse_vectors = getattr(sparse_vectors, "sparse_vectors", None)
        if isinstance(sparse_vectors, dict):
            return {str(name) for name in sparse_vectors.keys()}
        return set()

    def ensure_collection(self, vector_dims: Dict[str, int]):
        if not vector_dims:
            raise ValueError("At least one dense vector dimension must be provided.")
        if any(int(value) <= 0 for value in vector_dims.values()):
            raise ValueError("Vector dimensions must be greater than zero.")
        client = self._get_client()
        _, qmodels = _require_qdrant_client()
        collection_exists = client.collection_exists(self.collection_name)
        if collection_exists and self.drop_if_exists and not self._collection_ready:
            client.delete_collection(self.collection_name)
            collection_exists = False
        if not collection_exists:
            vectors_config = {
                name: self._build_dense_vector_params(dim, qmodels)
                for name, dim in vector_dims.items()
            }
            create_kwargs = {
                "collection_name": self.collection_name,
                "vectors_config": vectors_config,
                "on_disk_payload": self.on_disk_payload,
                "hnsw_config": qmodels.HnswConfigDiff(
                    m=self.hnsw_m,
                    ef_construct=self.hnsw_ef_construct,
                    on_disk=self.hnsw_on_disk,
                ),
            }
            sparse_vectors_config: Dict[str, Any] = {}
            if self.enable_sparse_vectors:
                sparse_vectors_config[self.sparse_vector_name] = qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
            if self.enable_bge_sparse:
                sparse_vectors_config[self.bge_sparse_vector_name] = qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
            if sparse_vectors_config:
                create_kwargs["sparse_vectors_config"] = sparse_vectors_config
            client.create_collection(**create_kwargs)
        else:
            collection_info = client.get_collection(self.collection_name)
            required_dense_names = set(vector_dims.keys())
            required_sparse_names = set()
            if self.enable_sparse_vectors:
                required_sparse_names.add(self.sparse_vector_name)
            if self.enable_bge_sparse:
                required_sparse_names.add(self.bge_sparse_vector_name)
            existing_dense_names = self._extract_existing_dense_vector_names(collection_info)
            existing_sparse_names = self._extract_existing_sparse_vector_names(collection_info)
            missing_dense = sorted(required_dense_names - existing_dense_names)
            missing_sparse = sorted(required_sparse_names - existing_sparse_names)
            if missing_dense or missing_sparse:
                missing_parts: List[str] = []
                if missing_dense:
                    missing_parts.append(f"dense={missing_dense}")
                if missing_sparse:
                    missing_parts.append(f"sparse={missing_sparse}")
                raise RuntimeError(
                    "Existing Qdrant collection is missing required vector fields for the current ingest configuration: "
                    + ", ".join(missing_parts)
                    + ". Re-run with --drop-if-exists once to recreate the collection."
                )
            try:
                client.update_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        name: self._build_dense_vector_diff(qmodels)
                        for name in required_dense_names
                    },
                    hnsw_config=qmodels.HnswConfigDiff(
                        m=self.hnsw_m,
                        ef_construct=self.hnsw_ef_construct,
                        on_disk=self.hnsw_on_disk,
                    ),
                    quantization_config=self._build_quantization_config(qmodels),
                )
            except Exception:
                pass
        if self.url:
            for field_name, schema in [
                ("chunk_uid", qmodels.PayloadSchemaType.KEYWORD),
                ("chunk_level", qmodels.PayloadSchemaType.KEYWORD),
                ("source_file", qmodels.PayloadSchemaType.KEYWORD),
                ("doc_title", qmodels.PayloadSchemaType.KEYWORD),
                ("chunk_type", qmodels.PayloadSchemaType.KEYWORD),
                ("parent_chunk_uid", qmodels.PayloadSchemaType.KEYWORD),
            ]:
                try:
                    client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema,
                        wait=True,
                    )
                except Exception:
                    pass
        self._vector_dims = {str(name): int(dim) for name, dim in vector_dims.items()}
        self._collection_ready = True

    def _resolve_vector_dims(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        resolved_dims: Dict[str, int] = {}
        for idx, chunk in enumerate(chunks):
            embedding = chunk.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"Chunk #{idx} does not contain a usable embedding vector.")
            if self.dense_vector_name not in resolved_dims:
                resolved_dims[self.dense_vector_name] = len(embedding)
            elif len(embedding) != resolved_dims[self.dense_vector_name]:
                raise ValueError(
                    f"Chunk #{idx} embedding dimension mismatch: expected {resolved_dims[self.dense_vector_name]}, got {len(embedding)}."
                )

            bge_dense_embedding = chunk.get("bge_dense_embedding")
            if self.enable_bge_dense and isinstance(bge_dense_embedding, list) and bge_dense_embedding:
                if self.bge_dense_vector_name not in resolved_dims:
                    resolved_dims[self.bge_dense_vector_name] = len(bge_dense_embedding)
                elif len(bge_dense_embedding) != resolved_dims[self.bge_dense_vector_name]:
                    raise ValueError(
                        "BGE-M3 embedding dimension mismatch: "
                        f"expected {resolved_dims[self.bge_dense_vector_name]}, got {len(bge_dense_embedding)}."
                    )
        if self.dense_vector_name not in resolved_dims:
            raise ValueError("None of the provided chunks contains a usable embedding vector.")
        return resolved_dims

    def _build_chunk_id(self, source_file: str, chunk: Dict[str, Any]) -> str:
        return resolve_chunk_uid(source_file, chunk)

    def _to_point(self, chunk: Dict[str, Any], source_file: str):
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Chunk does not contain a usable embedding vector.")
        _, qmodels = _require_qdrant_client()
        header_path = chunk.get("header_path", [])
        quality_flags = chunk.get("quality_flags", [])
        alias_terms = chunk.get("alias_terms", [])
        keywords = chunk.get("keywords", [])
        payload = {
            "chunk_uid": self._build_chunk_id(source_file, chunk),
            "chunk_level": str(chunk.get("chunk_level", "child") or "child"),
            "source_file": str(source_file),
            "source": str(chunk.get("source", source_file) or source_file),
            "source_file_name": str(chunk.get("source_file_name", Path(source_file).name)),
            "doc_id": int(chunk.get("doc_id", 0) or 0),
            "doc_title": str(chunk.get("doc_title", "") or ""),
            "doc_type": str(chunk.get("doc_type", "") or ""),
            "source_text_profile": str(chunk.get("source_text_profile", "") or ""),
            "section_id": int(chunk.get("section_id", 0) or 0),
            "chunk_in_section": int(chunk.get("chunk_in_section", 0) or 0),
            "section_title": str(chunk.get("section_title", "") or ""),
            "page_no": int(chunk.get("page_no", 0) or 0),
            "page_label": str(chunk.get("page_label", "") or ""),
            "section_kind": str(chunk.get("section_kind", "") or ""),
            "knowledge_unit_type": str(chunk.get("knowledge_unit_type", "") or ""),
            "header_path": list(header_path) if isinstance(header_path, list) else [str(header_path or "")],
            "header_path_text": str(chunk.get("header_path_text", "") or ""),
            "chunk_type": str(chunk.get("chunk_type", "") or ""),
            "semantic_role": str(chunk.get("semantic_role", "") or ""),
            "table_family": str(chunk.get("table_family", "") or ""),
            "text": str(chunk.get("text", "") or ""),
            "embedding_text": str(chunk.get("embedding_text", "") or ""),
            "retrieval_text": str(chunk.get("retrieval_text", "") or ""),
            "alias_terms": list(alias_terms) if isinstance(alias_terms, list) else [str(alias_terms or "")],
            "keywords": list(keywords) if isinstance(keywords, list) else [str(keywords or "")],
            "quality_flags": list(quality_flags) if isinstance(quality_flags, list) else [str(quality_flags or "")],
            "text_length": int(chunk.get("text_length", 0) or 0),
            "table_row_count": int(chunk.get("table_row_count", 0) or 0),
            "answerability_score": float(chunk.get("answerability_score", 0.0) or 0.0),
            "quality_score": float(chunk.get("quality_score", 0.0) or 0.0),
            "info_density": float(chunk.get("info_density", 0.0) or 0.0),
            "noise_score": float(chunk.get("noise_score", 0.0) or 0.0),
            "summary_consistency_score": float(chunk.get("summary_consistency_score", 0.0) or 0.0),
            "page_consistency_flags": list(chunk.get("page_consistency_flags", []) or []),
            "is_retrieval_eligible": bool(chunk.get("is_retrieval_eligible", False)),
            "parent_chunk_uid": str(chunk.get("parent_chunk_uid", "") or ""),
            "parent_chunk_index": int(chunk.get("parent_chunk_index", 0) or 0),
            "parent_chunk_count": int(chunk.get("parent_chunk_count", 0) or 0),
            "child_chunk_uids": list(chunk.get("child_chunk_uids", []) or []),
            "child_chunk_count": int(chunk.get("child_chunk_count", 0) or 0),
            "child_chunk_start_index": int(chunk.get("child_chunk_start_index", 0) or 0),
            "child_chunk_end_index": int(chunk.get("child_chunk_end_index", 0) or 0),
            "prev_chunk_uid": str(chunk.get("prev_chunk_uid", "") or ""),
            "next_chunk_uid": str(chunk.get("next_chunk_uid", "") or ""),
            "metadata_version": str(chunk.get("metadata_version", "stage2") or "stage2"),
            "bge_m3_ready": bool(chunk.get("bge_m3_ready", False)),
        }
        payload["header_path"] = [value for value in payload["header_path"] if str(value).strip()]
        payload["quality_flags"] = [value for value in payload["quality_flags"] if str(value).strip()]
        payload["page_consistency_flags"] = [value for value in payload["page_consistency_flags"] if str(value).strip()]
        payload["alias_terms"] = [value for value in payload["alias_terms"] if str(value).strip()]
        payload["keywords"] = [value for value in payload["keywords"] if str(value).strip()]
        payload["child_chunk_uids"] = [value for value in payload["child_chunk_uids"] if str(value).strip()]
        vector: Dict[str, Any] = {self.dense_vector_name: embedding}
        bge_dense_embedding = chunk.get("bge_dense_embedding")
        if self.enable_bge_dense and isinstance(bge_dense_embedding, list) and bge_dense_embedding:
            vector[self.bge_dense_vector_name] = bge_dense_embedding
        if self.enable_sparse_vectors:
            sparse_vector = chunk.get("sparse_vector", {}) or {}
            indices = [int(value) for value in sparse_vector.get("indices", [])]
            values = [float(value) for value in sparse_vector.get("values", [])]
            if indices and values and len(indices) == len(values):
                vector[self.sparse_vector_name] = qmodels.SparseVector(indices=indices, values=values)
        if self.enable_bge_sparse:
            bge_sparse_vector = chunk.get("bge_sparse_vector", {}) or {}
            indices = [int(value) for value in bge_sparse_vector.get("indices", [])]
            values = [float(value) for value in bge_sparse_vector.get("values", [])]
            if indices and values and len(indices) == len(values):
                vector[self.bge_sparse_vector_name] = qmodels.SparseVector(indices=indices, values=values)
        return qmodels.PointStruct(
            id=self._build_chunk_id(source_file, chunk),
            vector=vector,
            payload=payload,
        )

    def upsert_chunks(self, chunks: List[Dict[str, Any]], source_file: str) -> int:
        if not chunks:
            return 0
        vector_dims = self._resolve_vector_dims(chunks)
        self.ensure_collection(vector_dims)
        points = [self._to_point(chunk, source_file) for chunk in chunks]
        total = 0
        client = self._get_client()
        total_batches = max(1, math.ceil(len(points) / DEFAULT_UPSERT_BATCH_SIZE))
        for start in range(0, len(points), DEFAULT_UPSERT_BATCH_SIZE):
            batch = points[start:start + DEFAULT_UPSERT_BATCH_SIZE]
            batch_index = start // DEFAULT_UPSERT_BATCH_SIZE + 1
            batch_started_at = time.perf_counter()
            client.upsert(collection_name=self.collection_name, points=batch, wait=self.upsert_wait)
            total += len(batch)
            elapsed = time.perf_counter() - batch_started_at
            print(
                f"[INFO] Qdrant upsert batch {batch_index}/{total_batches}: "
                f"{total}/{len(points)} points in {elapsed:.1f}s (wait={self.upsert_wait})"
            )
        return total


def sleep_with_jitter(base_delay: float):
    if base_delay <= 0:
        return
    jittered_delay = random.uniform(base_delay * 0.8, base_delay * 1.2)
    time.sleep(max(0.0, jittered_delay))


def call_embedding_api_with_retry(
    embedder: QwenLocalEmbeddingModel,
    texts: List[str],
    batch_size: int,
    max_retries: int,
    initial_delay: float,
) -> List[List[float]]:
    if not texts:
        return []
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return embedder.encode(texts, batch_size=batch_size)
        except Exception as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                raise RuntimeError(f"Local embedding request failed after internal batch downscaling: {exc}") from exc
            if attempt == max_retries:
                raise RuntimeError(f"Local embedding request failed: {exc}") from exc
            print(f"[WARN] Local embedding batch failed, retry {attempt}/{max_retries} after {delay:.1f}s: {message}")
            sleep_with_jitter(delay)
            delay *= 2
    raise RuntimeError("Local embedding request exhausted unexpectedly")


def iter_chunk_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.name.endswith(".chunks.json"):
            return [input_path]
        raise FileNotFoundError(f"Input file is not a .chunks.json file: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Chunk input path does not exist: {input_path}")
    chunk_files = [path for path in input_path.rglob("*.chunks.json") if path.name != "_index.json"]
    chunk_files.sort()
    return chunk_files


def load_chunk_payload(chunk_file: Path) -> Dict[str, Any]:
    payload = json.loads(chunk_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid chunk file format: {chunk_file}")
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"Invalid chunk list in: {chunk_file}")
    parent_chunks = payload.get("parent_chunks", [])
    if not isinstance(parent_chunks, list):
        raise ValueError(f"Invalid parent chunk list in: {chunk_file}")
    payload["chunks"] = chunks
    payload["parent_chunks"] = parent_chunks
    return payload

def build_file_fingerprint(path: Path) -> Dict[str, Any]:
    hasher = hashlib.sha1()
    total_size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
            total_size += len(block)
    stat = path.stat()
    return {
        "sha1": hasher.hexdigest(),
        "size": total_size,
        "mtime_ns": stat.st_mtime_ns,
    }

def embedded_payload_matches_source(
    embedded_output_file: Path,
    chunk_file: Path,
    required_embed_scope: str = "qdrant",
    require_bge_m3: bool = False,
) -> tuple[bool, Dict[str, Any] | None, Dict[str, Any]]:
    current_fingerprint = build_file_fingerprint(chunk_file)
    if not embedded_output_file.exists():
        return False, None, current_fingerprint

    try:
        payload = load_chunk_payload(embedded_output_file)
    except Exception:
        return False, None, current_fingerprint

    def scope_is_sufficient() -> bool:
        if required_embed_scope != "all":
            base_scope_ok = True
        else:
            payload_scope = str(payload.get("embedding_scope") or ("all" if payload.get("embedding_ready") else "")).strip().lower()
            base_scope_ok = payload_scope == "all" and chunks_have_embeddings(get_all_payload_chunks(payload))
        if not base_scope_ok:
            return False
        if require_bge_m3:
            return chunks_have_bge_artifacts(get_all_payload_chunks(payload))
        return True

    embedded_fp = payload.get("source_fingerprint")
    if isinstance(embedded_fp, dict) and embedded_fp.get("sha1") == current_fingerprint["sha1"]:
        return (True, payload, current_fingerprint) if scope_is_sufficient() else (False, payload, current_fingerprint)

    if not embedded_fp:
        try:
            if embedded_output_file.stat().st_mtime_ns >= chunk_file.stat().st_mtime_ns:
                return (True, payload, current_fingerprint) if scope_is_sufficient() else (False, payload, current_fingerprint)
        except FileNotFoundError:
            return False, None, current_fingerprint

    return False, payload, current_fingerprint

def build_embedded_output_path(chunk_file: Path, input_root: Path, output_root: Path) -> Path:
    relative_parent = Path() if input_root.is_file() else chunk_file.parent.relative_to(input_root)
    return output_root / relative_parent / f"{chunk_file.stem}.embedded.json"

def get_all_payload_chunks(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(payload.get("chunks", [])) + list(payload.get("parent_chunks", []))


def chunks_have_embeddings(chunks: List[Dict[str, Any]]) -> bool:
    return bool(chunks) and all(isinstance(item.get("embedding"), list) and item.get("embedding") for item in chunks)


def chunks_have_bge_artifacts(chunks: List[Dict[str, Any]]) -> bool:
    return bool(chunks) and all(
        isinstance(item.get("bge_dense_embedding"), list)
        and bool(item.get("bge_dense_embedding"))
        and isinstance(item.get("bge_sparse_vector"), dict)
        and bool((item.get("bge_sparse_vector") or {}).get("indices"))
        for item in chunks
    )


def select_embedding_targets(
    all_chunks: List[Dict[str, Any]],
    store: Optional["QdrantLocalVectorStore"],
    embed_scope: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    if store is None:
        return all_chunks, [], 0
    qdrant_chunks, skipped_count = filter_chunks_for_qdrant(all_chunks)
    if embed_scope == "all":
        return all_chunks, qdrant_chunks, skipped_count
    return qdrant_chunks, qdrant_chunks, skipped_count


def ensure_embeddings_for_chunks(
    chunks: List[Dict[str, Any]],
    embedder: QwenLocalEmbeddingModel,
    batch_size: int,
    max_retries: int,
    initial_delay: float,
    request_interval: float,
) -> List[Dict[str, Any]]:
    if not chunks:
        return chunks
    missing_indexes: List[int] = []
    pending_texts: List[str] = []
    seen_texts: set[str] = set()
    embedding_cache: Dict[str, List[float]] = {}

    for idx, chunk in enumerate(chunks):
        embedding = chunk.get("embedding")
        if isinstance(embedding, list) and embedding:
            continue
        text = (chunk.get("embedding_text") or chunk.get("text") or "").strip()
        if not text:
            raise ValueError(f"Chunk #{idx} has no embedding_text/text and cannot be vectorized.")
        missing_indexes.append(idx)
        if text not in seen_texts:
            pending_texts.append(text)
            seen_texts.add(text)

    if not missing_indexes:
        return chunks

    print(
        f"[INFO] Embedding {len(pending_texts)} unique texts for {len(missing_indexes)} chunks "
        f"(batch_size={batch_size}, max_length={embedder.max_length})"
    )
    for start in range(0, len(pending_texts), batch_size):
        batch = pending_texts[start:start + batch_size]
        try:
            batch_started_at = time.perf_counter()
            batch_embeddings = call_embedding_api_with_retry(embedder, batch, batch_size, max_retries, initial_delay)
            elapsed = time.perf_counter() - batch_started_at
            print(f"[INFO] Embedded batch {min(start + len(batch), len(pending_texts))}/{len(pending_texts)} in {elapsed:.1f}s")
        except Exception as exc:
            preview = batch[0][:120] if batch else ""
            raise RuntimeError(
                f"Failed to embed batch starting at pending_text index {start} "
                f"(batch_size={len(batch)}, first_text_preview={preview!r}): {exc}"
            ) from exc
        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding batch size mismatch at pending_text index {start}: "
                f"expected {len(batch)} vectors, got {len(batch_embeddings)}."
            )
        for text, embedding in zip(batch, batch_embeddings):
            embedding_cache[text] = embedding
        if request_interval > 0 and start + batch_size < len(pending_texts):
            time.sleep(request_interval)

    for idx in missing_indexes:
        text = (chunks[idx].get("embedding_text") or chunks[idx].get("text") or "").strip()
        if text not in embedding_cache:
            preview = text[:160]
            raise RuntimeError(f"Missing cached embedding when backfilling chunk #{idx}. text_preview={preview!r}")
        chunks[idx]["embedding"] = embedding_cache[text]
    return chunks


def write_embedded_payload(
    output_file: Path,
    payload: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    parent_chunks: List[Dict[str, Any]],
    embed_scope: str = "all",
    embedded_target_count: int | None = None,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    new_payload = dict(payload)
    all_chunks = chunks + parent_chunks
    embedded_count = int(
        embedded_target_count if embedded_target_count is not None else sum(1 for item in all_chunks if isinstance(item.get("embedding"), list) and item.get("embedding"))
    )
    new_payload["chunk_count"] = len(chunks)
    new_payload["parent_chunk_count"] = len(parent_chunks)
    new_payload["embedding_ready"] = embedded_count > 0 or not all_chunks
    new_payload["all_embeddings_ready"] = chunks_have_embeddings(all_chunks)
    new_payload["embedding_scope"] = embed_scope
    new_payload["embedded_target_count"] = embedded_count
    new_payload["chunks"] = chunks
    new_payload["parent_chunks"] = parent_chunks
    output_file.write_text(json.dumps(new_payload, ensure_ascii=False), encoding="utf-8")


def safe_console_text(value: Any) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except Exception:
        return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def print_preview(chunks: List[Dict[str, Any]], top_k: int = 2):
    print(f"Total chunks: {len(chunks)}")
    print("=" * 100)
    for item in chunks[:top_k]:
        header_path = item.get("header_path", [])
        if isinstance(header_path, list):
            header_path_text = " > ".join(header_path) if header_path else "(none)"
        else:
            header_path_text = str(header_path or "(none)")
        embedding = item.get("embedding") or []
        print(f"doc_id: {safe_console_text(item.get('doc_id', ''))}")
        print(f"doc_title: {safe_console_text(item.get('doc_title', ''))}")
        print(f"section_id: {safe_console_text(item.get('section_id', ''))}")
        print(f"chunk_in_section: {safe_console_text(item.get('chunk_in_section', ''))}")
        print(f"chunk_type: {safe_console_text(item.get('chunk_type', 'paragraph'))}")
        print(f"header_path: {safe_console_text(header_path_text)}")
        print(f"keywords: {safe_console_text(item.get('keywords', []))}")
        print(f"text_length: {safe_console_text(item.get('text_length', 0))}")
        print(f"text_preview: {safe_console_text((item.get('text') or '')[:260])}")
        print(f"embedding_dim: {safe_console_text(len(embedding))}")
        print("-" * 100)


def preflight_runtime_checks(
    chunk_files: List[Path],
    model_path: str,
    device: str,
    require_qdrant: bool,
    bge_m3_model_path: str = "",
) -> None:
    if not chunk_files:
        raise FileNotFoundError("No .chunks.json files found for embedding.")
    issues = collect_runtime_dependency_issues(
        model_name_or_path=model_path,
        device=device,
        require_qdrant=require_qdrant,
        bge_m3_model_path=bge_m3_model_path,
        require_bge_m3=bool(str(bge_m3_model_path or "").strip()),
    )
    if not issues:
        return
    details = "\n".join(f"- {issue}" for issue in issues)
    raise RuntimeError(
        "启动前检查失败，当前环境还不满足向量化运行条件:\n"
        f"{details}\n"
        "处理建议:\n"
        f"- 先完成本地 embedding 依赖安装。\n"
        f"- 当前 embedding 模型路径配置是: {model_path}\n"
        f"- 待处理 chunk 文件数量: {len(chunk_files)}"
    )


def process_one_chunk_file(
    chunk_file: Path,
    input_root: Path,
    output_root: Path,
    embedder: QwenLocalEmbeddingModel,
    store: Optional[QdrantLocalVectorStore],
    bge_helper: Optional[BgeM3IngestHelper],
    batch_size: int,
    max_retries: int,
    initial_delay: float,
    request_interval: float,
    write_json: bool,
    force: bool,
    reupsert_existing: bool,
    enrich_metadata: bool,
    embed_scope: str,
    preview_top_k: int,
) -> Dict[str, Any]:
    file_started_at = time.perf_counter()
    embed_seconds = 0.0
    qdrant_seconds = 0.0
    json_seconds = 0.0
    embedded_output_file = build_embedded_output_path(chunk_file, input_root, output_root)
    required_embed_scope = embed_scope if store is not None else "all"

    if not force:
        embedded_matches_source, payload, current_fingerprint = embedded_payload_matches_source(
            embedded_output_file,
            chunk_file,
            required_embed_scope=required_embed_scope,
            require_bge_m3=bool(bge_helper and bge_helper.enabled),
        )
        if embedded_matches_source and payload is not None:
            if "source_fingerprint" not in payload or not payload.get("source_fingerprint"):
                payload["source_fingerprint"] = current_fingerprint
                payload["source_chunk_file"] = str(chunk_file)
                if write_json:
                    write_embedded_payload(embedded_output_file, payload, payload["chunks"], payload["parent_chunks"])
            chunks = payload["chunks"]
            parent_chunks = payload["parent_chunks"]
            qdrant_count = 0
            skipped_low_quality_count = 0
            status = "skipped"
            if store is not None and reupsert_existing:
                source_file = str(payload.get("source_file") or chunk_file)
                ensure_chunk_uids(chunks, source_file)
                ensure_chunk_uids(parent_chunks, source_file)
                if enrich_metadata:
                    chunks = [enrich_chunk_metadata(chunk, source_file) for chunk in chunks]
                    parent_chunks = [enrich_chunk_metadata(chunk, source_file) for chunk in parent_chunks]
                if bge_helper is not None:
                    chunks = bge_helper.ensure_artifacts_for_chunks(chunks, max_retries, initial_delay, request_interval)
                    parent_chunks = bge_helper.ensure_artifacts_for_chunks(parent_chunks, max_retries, initial_delay, request_interval)
                    chunks = [bge_helper.enrich_chunk(chunk) for chunk in chunks]
                    parent_chunks = [bge_helper.enrich_chunk(chunk) for chunk in parent_chunks]
                all_chunks = chunks + parent_chunks
                embedding_targets, qdrant_chunks, skipped_low_quality_count = select_embedding_targets(all_chunks, store, embed_scope)
                if not chunks_have_embeddings(embedding_targets):
                    embed_started_at = time.perf_counter()
                    ensure_embeddings_for_chunks(embedding_targets, embedder, batch_size, max_retries, initial_delay, request_interval)
                    embed_seconds += time.perf_counter() - embed_started_at
                qdrant_started_at = time.perf_counter()
                qdrant_count = store.upsert_chunks(qdrant_chunks, source_file=source_file)
                qdrant_seconds += time.perf_counter() - qdrant_started_at
                status = "reused"
            total_seconds = time.perf_counter() - file_started_at
            print(
                f"[INFO] {status.upper()} {chunk_file.name}: "
                f"targets={len(chunks) + len(parent_chunks)} embed_scope={required_embed_scope} "
                f"in {total_seconds:.1f}s (embed={embed_seconds:.1f}s, qdrant={qdrant_seconds:.1f}s)"
            )
            return {
                "source_chunk_file": str(chunk_file),
                "embedded_output_file": str(embedded_output_file),
                "status": status,
                "chunk_count": len(chunks),
                "parent_chunk_count": len(parent_chunks),
                "embed_scope": required_embed_scope,
                "embed_seconds": embed_seconds,
                "qdrant_seconds": qdrant_seconds,
                "json_seconds": json_seconds,
                "total_seconds": total_seconds,
                "qdrant_point_count": qdrant_count,
                "skipped_low_quality_count": skipped_low_quality_count,
                "source_fingerprint": current_fingerprint,
            }

    payload = load_chunk_payload(chunk_file)
    chunks = payload["chunks"]
    parent_chunks = payload["parent_chunks"]
    source_file = str(payload.get("source_file") or chunk_file)
    ensure_chunk_uids(chunks, source_file)
    ensure_chunk_uids(parent_chunks, source_file)
    if enrich_metadata:
        chunks = [enrich_chunk_metadata(chunk, source_file) for chunk in chunks]
        parent_chunks = [enrich_chunk_metadata(chunk, source_file) for chunk in parent_chunks]
    if bge_helper is not None:
        chunks = bge_helper.ensure_artifacts_for_chunks(chunks, max_retries, initial_delay, request_interval)
        parent_chunks = bge_helper.ensure_artifacts_for_chunks(parent_chunks, max_retries, initial_delay, request_interval)
        chunks = [bge_helper.enrich_chunk(chunk) for chunk in chunks]
        parent_chunks = [bge_helper.enrich_chunk(chunk) for chunk in parent_chunks]
    all_chunks = chunks + parent_chunks
    embedding_targets, qdrant_chunks, skipped_low_quality_count = select_embedding_targets(all_chunks, store, embed_scope)
    embed_started_at = time.perf_counter()
    ensure_embeddings_for_chunks(embedding_targets, embedder, batch_size, max_retries, initial_delay, request_interval)
    embed_seconds += time.perf_counter() - embed_started_at

    if write_json:
        payload["source_chunk_file"] = str(chunk_file)
        payload["source_fingerprint"] = build_file_fingerprint(chunk_file)
        json_started_at = time.perf_counter()
        write_embedded_payload(
            embedded_output_file,
            payload,
            chunks,
            parent_chunks,
            embed_scope=embed_scope if store is not None else "all",
            embedded_target_count=len(embedding_targets),
        )
        json_seconds += time.perf_counter() - json_started_at

    qdrant_count = 0
    if store is not None:
        qdrant_started_at = time.perf_counter()
        qdrant_count = store.upsert_chunks(qdrant_chunks, source_file=source_file)
        qdrant_seconds += time.perf_counter() - qdrant_started_at

    print(f"[INFO] Embedded {len(chunks)} child chunks and {len(parent_chunks)} parent chunks from {chunk_file.name}")
    if skipped_low_quality_count:
        print(f"[INFO] Skipped {skipped_low_quality_count} low-quality chunks before Qdrant upsert for {chunk_file.name}")
    if preview_top_k > 0:
        print_preview(chunks, top_k=preview_top_k)
    total_seconds = time.perf_counter() - file_started_at
    print(
        f"[INFO] File timing for {chunk_file.name}: "
        f"total={total_seconds:.1f}s embed={embed_seconds:.1f}s json={json_seconds:.1f}s qdrant={qdrant_seconds:.1f}s "
        f"embedded_targets={len(embedding_targets)} qdrant_points={qdrant_count}"
    )
    return {
        "source_chunk_file": str(chunk_file),
        "embedded_output_file": str(embedded_output_file),
        "status": "processed",
        "chunk_count": len(chunks),
        "parent_chunk_count": len(parent_chunks),
        "embed_scope": required_embed_scope,
        "embedded_target_count": len(embedding_targets),
        "embed_seconds": embed_seconds,
        "qdrant_seconds": qdrant_seconds,
        "json_seconds": json_seconds,
        "total_seconds": total_seconds,
        "qdrant_point_count": qdrant_count,
        "skipped_low_quality_count": skipped_low_quality_count,
        "source_fingerprint": payload.get("source_fingerprint", {}),
    }

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vectorize .chunks.json files and optionally sync them to local Qdrant.")
    parser.add_argument("--input-path", default=str(DEFAULT_CHUNK_INPUT), help="Chunk file or directory containing .chunks.json files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_EMBED_OUTPUT), help="Directory used to write *.embedded.json files.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen embedding model path.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Embedding device, e.g. auto/cpu/cuda:0.")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, help="Embedding dtype, e.g. float16/float32/bfloat16.")
    parser.add_argument("--attn-implementation", default=DEFAULT_ATTN_IMPL, help="Transformers attention implementation.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Tokenizer max length.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retries for a failed embedding batch.")
    parser.add_argument("--initial-delay", type=float, default=DEFAULT_INITIAL_DELAY, help="Initial retry delay in seconds.")
    parser.add_argument("--request-interval", type=float, default=DEFAULT_REQUEST_INTERVAL, help="Delay between embedding batches in seconds.")
    parser.add_argument("--db-path", default=str(DEFAULT_QDRANT_PATH), help="Local Qdrant storage directory, or ':memory:' for in-memory local mode.")
    parser.add_argument("--url", default=DEFAULT_QDRANT_URL, help="Qdrant server URL, e.g. http://localhost:6333.")
    parser.add_argument("--api-key", default=DEFAULT_QDRANT_API_KEY, help="Optional Qdrant API key.")
    parser.add_argument("--prefer-grpc", action="store_true", default=DEFAULT_QDRANT_PREFER_GRPC, help="Prefer gRPC when using a Qdrant server URL.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--dense-on-disk", dest="dense_on_disk", action="store_true", default=DEFAULT_QDRANT_DENSE_ON_DISK, help="Store dense vectors on disk instead of keeping them in RAM.")
    parser.add_argument("--no-dense-on-disk", dest="dense_on_disk", action="store_false", help="Keep dense vectors in RAM for lower-latency search.")
    parser.add_argument("--hnsw-on-disk", dest="hnsw_on_disk", action="store_true", default=DEFAULT_QDRANT_HNSW_ON_DISK, help="Store the HNSW graph on disk.")
    parser.add_argument("--no-hnsw-on-disk", dest="hnsw_on_disk", action="store_false", help="Keep the HNSW graph in RAM for lower-latency search.")
    parser.add_argument("--hnsw-m", type=int, default=DEFAULT_QDRANT_HNSW_M, help="HNSW graph degree. Use 32+ for stronger recall on larger collections.")
    parser.add_argument("--hnsw-ef-construct", type=int, default=DEFAULT_QDRANT_HNSW_EF_CONSTRUCT, help="HNSW build ef_construct. Use 256+ for stronger recall on larger collections.")
    parser.add_argument("--on-disk-payload", dest="on_disk_payload", action="store_true", default=DEFAULT_QDRANT_ON_DISK_PAYLOAD, help="Store payload data on disk.")
    parser.add_argument("--no-on-disk-payload", dest="on_disk_payload", action="store_false", help="Keep payload data in RAM when possible.")
    parser.add_argument("--scalar-quantization", dest="enable_scalar_quantization", action="store_true", default=DEFAULT_QDRANT_ENABLE_SCALAR_QUANTIZATION, help="Enable int8 scalar quantization for dense vectors.")
    parser.add_argument("--no-scalar-quantization", dest="enable_scalar_quantization", action="store_false", help="Disable scalar quantization for dense vectors.")
    parser.add_argument("--scalar-quantization-always-ram", dest="scalar_quantization_always_ram", action="store_true", default=DEFAULT_QDRANT_SCALAR_QUANTIZATION_ALWAYS_RAM, help="Keep quantized vectors in RAM for lower latency.")
    parser.add_argument("--no-scalar-quantization-always-ram", dest="scalar_quantization_always_ram", action="store_false", help="Allow quantized vectors to spill out of RAM.")
    parser.add_argument("--qdrant-upsert-wait", dest="qdrant_upsert_wait", action="store_true", default=DEFAULT_QDRANT_UPSERT_WAIT, help="Wait for each Qdrant upsert batch to be fully applied before continuing.")
    parser.add_argument("--no-qdrant-upsert-wait", dest="qdrant_upsert_wait", action="store_false", help="Do not wait for each Qdrant upsert batch; faster for server deployments.")
    parser.add_argument("--force", action="store_true", default=DEFAULT_FORCE_REPROCESS, help="Recompute even if embedded output already exists.")
    parser.add_argument("--reupsert-existing", action="store_true", default=DEFAULT_REUPSERT_EXISTING, help="When embedded JSON already exists, still re-upsert those chunks into Qdrant.")
    parser.add_argument("--drop-if-exists", action="store_true", default=DEFAULT_DROP_IF_EXISTS, help="Drop the Qdrant collection once before the first write.")
    parser.add_argument("--write-json", dest="write_json", action="store_true", default=DEFAULT_WRITE_EMBEDDED_JSON, help="Write *.embedded.json files for debugging.")
    parser.add_argument("--no-write-json", dest="write_json", action="store_false", help="Skip writing *.embedded.json files.")
    parser.add_argument("--store-to-qdrant", dest="store_to_qdrant", action="store_true", default=DEFAULT_STORE_TO_QDRANT, help="Sync vectors into local Qdrant.")
    parser.add_argument("--no-qdrant", dest="store_to_qdrant", action="store_false", help="Skip Qdrant sync and only do vectorization.")
    parser.add_argument("--enrich-metadata", dest="enrich_metadata", action="store_true", default=DEFAULT_ENABLE_METADATA_ENRICH, help="Enrich chunks with retrieval-oriented metadata.")
    parser.add_argument("--no-enrich-metadata", dest="enrich_metadata", action="store_false", help="Disable stage-2 metadata enrichment.")
    parser.add_argument("--embed-scope", choices=["qdrant", "all"], default=DEFAULT_EMBED_SCOPE, help="Vectorize only chunks that will be stored in Qdrant, or vectorize all chunks for full JSON cache.")
    parser.add_argument("--preview-top-k", type=int, default=DEFAULT_PREVIEW_TOP_K, help="How many chunk previews to print per processed file.")
    parser.add_argument("--bge-m3-model-path", default=DEFAULT_BGE_M3_MODEL_PATH, help="Optional local BGE-M3 model path used to build extra dense/sparse retrieval vectors.")
    parser.add_argument("--bge-m3-device", default=DEFAULT_BGE_M3_DEVICE, help="BGE-M3 device, e.g. auto/cpu/cuda:0.")
    parser.add_argument("--bge-m3-batch-size", type=int, default=DEFAULT_BGE_M3_BATCH_SIZE, help="BGE-M3 encoding batch size.")
    parser.add_argument("--bge-m3-query-max-length", type=int, default=DEFAULT_BGE_M3_QUERY_MAX_LENGTH, help="BGE-M3 max query length.")
    parser.add_argument("--bge-m3-passage-max-length", type=int, default=DEFAULT_BGE_M3_PASSAGE_MAX_LENGTH, help="BGE-M3 max passage length.")
    parser.add_argument("--bge-m3-use-fp16", dest="bge_m3_use_fp16", action="store_true", default=DEFAULT_BGE_M3_USE_FP16, help="Use fp16 for BGE-M3 when CUDA is available.")
    parser.add_argument("--no-bge-m3-use-fp16", dest="bge_m3_use_fp16", action="store_false", help="Disable fp16 for BGE-M3.")
    return parser

def main() -> int:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    chunk_files = iter_chunk_files(input_path)
    preflight_runtime_checks(
        chunk_files,
        args.model_path,
        args.device,
        args.store_to_qdrant,
        bge_m3_model_path=args.bge_m3_model_path,
    )

    embedder = QwenLocalEmbeddingModel(
        model_name_or_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
    )

    bge_helper = BgeM3IngestHelper(
        model_path=args.bge_m3_model_path,
        device=args.bge_m3_device,
        batch_size=args.bge_m3_batch_size,
        query_max_length=args.bge_m3_query_max_length,
        passage_max_length=args.bge_m3_passage_max_length,
        use_fp16=args.bge_m3_use_fp16,
    )

    store = None
    if args.store_to_qdrant:
        store = QdrantLocalVectorStore(
            db_path=args.db_path,
            collection_name=args.collection,
            drop_if_exists=args.drop_if_exists,
            url=args.url,
            api_key=args.api_key,
            prefer_grpc=args.prefer_grpc,
            enable_bge_dense=bool(bge_helper.enabled),
            enable_bge_sparse=bool(bge_helper.enabled),
            dense_on_disk=args.dense_on_disk,
            hnsw_on_disk=args.hnsw_on_disk,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construct=args.hnsw_ef_construct,
            on_disk_payload=args.on_disk_payload,
            enable_scalar_quantization=args.enable_scalar_quantization,
            scalar_quantization_always_ram=args.scalar_quantization_always_ram,
            upsert_wait=args.qdrant_upsert_wait,
        )

    processed = 0
    reused = 0
    skipped = 0
    failed = 0
    index: List[Dict[str, Any]] = []

    try:
        for chunk_file in chunk_files:
            try:
                summary = process_one_chunk_file(
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
                    force=args.force,
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
    total_embed_seconds = sum(float(item.get("embed_seconds", 0.0) or 0.0) for item in index if isinstance(item, dict))
    total_qdrant_seconds = sum(float(item.get("qdrant_seconds", 0.0) or 0.0) for item in index if isinstance(item, dict))
    total_json_seconds = sum(float(item.get("json_seconds", 0.0) or 0.0) for item in index if isinstance(item, dict))
    total_runtime_seconds = sum(float(item.get("total_seconds", 0.0) or 0.0) for item in index if isinstance(item, dict))

    print(f"\nDone. processed={processed}, reused={reused}, skipped={skipped}, failed={failed}")
    print(
        f"[INFO] Timing summary: total={total_runtime_seconds:.1f}s "
        f"embed={total_embed_seconds:.1f}s json={total_json_seconds:.1f}s qdrant={total_qdrant_seconds:.1f}s"
    )
    if args.write_json:
        print(f"[INFO] Embedded JSON files stored under: {output_dir}")
    print(f"[INFO] Embedding index written to: {index_path}")
    if args.store_to_qdrant:
        if args.url:
            print(f"[INFO] Qdrant URL: {args.url}")
        elif args.db_path == ":memory:":
            print("[INFO] Qdrant local mode: :memory:")
        else:
            print(f"[INFO] Local Qdrant path: {args.db_path}")
        print(f"[INFO] Qdrant collection: {args.collection}")
        print(
            f"[INFO] Qdrant tuning: dense_on_disk={args.dense_on_disk} "
            f"hnsw_on_disk={args.hnsw_on_disk} hnsw_m={args.hnsw_m} "
            f"hnsw_ef_construct={args.hnsw_ef_construct} on_disk_payload={args.on_disk_payload} "
            f"scalar_quantization={args.enable_scalar_quantization} "
            f"quantization_always_ram={args.scalar_quantization_always_ram}"
        )
        print(f"[INFO] Qdrant upsert wait: {args.qdrant_upsert_wait}")
    print(f"[INFO] Preview top-k: {args.preview_top_k}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
