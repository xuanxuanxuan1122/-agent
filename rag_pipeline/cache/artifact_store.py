from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rag_pipeline.runtime_cache import json_safe_default

from .artifact_models import (
    ARTIFACT_LEDGER_SCHEMA_VERSION,
    ArtifactWriteResult,
    as_dict,
    as_list,
)
from .artifact_paths import (
    artifact_ledger_path,
    artifact_object_root,
    env_flag,
    env_int,
    safe_path_part,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_safe_default)


def _json_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=json_safe_default)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_payload(value: Any) -> str:
    return _hash_text(_json_dumps(value))


def _decode_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _compact_text(value: Any, max_chars: int = 1000) -> str:
    text = str(value or "").strip()
    return text[:max_chars]


def _enforced_claim_status(
    status: str,
    *,
    facts: Sequence[Any],
    sources: Sequence[Any],
    claim_text: str,
    requirement_ids: Sequence[Any],
) -> str:
    """P5 integrity floor for claim_units.

    A claim is only ``validated`` when it has real evidence backing (>=1 fact AND
    >=1 source) AND actual claim text. ``requirement_ids`` are intentionally NOT
    part of the floor: the upstream evidence-id granularity gap (line-level
    ``EV-04-L22`` cited while requirements bind ``EV-04-22``) leaves them
    legitimately empty, so a missing requirement only downgrades to ``directional``
    — it never rejects an otherwise evidence-backed claim.
    """
    if str(status).strip().lower() != "validated":
        return status
    if not (list(facts) and list(sources) and str(claim_text).strip()):
        return "unsupported"
    if not list(requirement_ids):
        return "directional"
    return "validated"


def _enforced_section_status(status: str, *, claim_ids: Sequence[Any], used_fact_refs: Sequence[Any]) -> str:
    """P5 integrity floor for sections: only ``validated`` when the section
    actually references a claim or a fact. An empty section must not be
    ``validated`` so the writer/context-view never renders it as long body."""
    if str(status).strip().lower() != "validated":
        return status
    if not (list(claim_ids) or list(used_fact_refs)):
        return "unsupported"
    return status


class ArtifactStore:
    def __init__(
        self,
        path: Optional[Path] = None,
        object_root: Optional[Path] = None,
        *,
        inline_max_bytes: Optional[int] = None,
    ) -> None:
        self.path = path or artifact_ledger_path()
        self.object_root = object_root or artifact_object_root()
        self.inline_max_bytes = (
            inline_max_bytes
            if inline_max_bytes is not None
            else env_int("ARTIFACT_LEDGER_INLINE_MAX_BYTES", 64 * 1024, min_value=1024, max_value=10 * 1024 * 1024)
        )
        self._init_lock = threading.Lock()
        self._initialized = False

    def enabled(self) -> bool:
        return env_flag("ARTIFACT_LEDGER_ENABLED", True)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        journal_mode = str(os.getenv("ARTIFACT_LEDGER_SQLITE_JOURNAL_MODE", "WAL") or "WAL").strip().upper()
        if journal_mode not in {"WAL", "MEMORY", "OFF", "DELETE", "TRUNCATE", "PERSIST"}:
            journal_mode = "WAL"
        try:
            conn.execute(f"PRAGMA journal_mode={journal_mode}")
        except sqlite3.OperationalError:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def sqlite_journal_mode(self) -> str:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            return str(row[0] if row else "")

    def _ensure_schema(self) -> None:
        if self._initialized or not self.enabled():
            return
        with self._init_lock:
            if self._initialized:
                return
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        query TEXT NOT NULL DEFAULT '',
                        report_type TEXT,
                        status TEXT NOT NULL,
                        freshness_policy_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS evidence_requirements (
                        run_id TEXT NOT NULL,
                        requirement_id TEXT NOT NULL,
                        chapter_id TEXT,
                        hypothesis_id TEXT,
                        proof_role TEXT NOT NULL,
                        required_fields_json TEXT,
                        min_source_level TEXT,
                        claim_strength_ceiling TEXT,
                        freshness_required INTEGER DEFAULT 0,
                        max_cache_age_hours INTEGER,
                        status TEXT NOT NULL,
                        missing_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, requirement_id)
                    );

                    CREATE TABLE IF NOT EXISTS sources (
                        canonical_source_id TEXT PRIMARY KEY,
                        canonical_url TEXT,
                        title TEXT,
                        publisher TEXT,
                        published_at TEXT,
                        fetched_at TEXT,
                        source_type TEXT,
                        source_level TEXT,
                        verification_status TEXT,
                        content_hash TEXT,
                        storage_uri TEXT,
                        status TEXT NOT NULL,
                        payload_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS run_sources (
                        run_id TEXT NOT NULL,
                        run_source_id TEXT NOT NULL,
                        canonical_source_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, run_source_id)
                    );

                    CREATE TABLE IF NOT EXISTS artifacts (
                        artifact_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        artifact_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        requirement_id TEXT,
                        source_id TEXT,
                        fact_id TEXT,
                        claim_id TEXT,
                        section_id TEXT,
                        schema_version TEXT,
                        prompt_version TEXT,
                        model TEXT,
                        producer_version TEXT,
                        input_hash TEXT,
                        output_hash TEXT,
                        content_hash TEXT,
                        storage_uri TEXT,
                        storage_bytes INTEGER DEFAULT 0,
                        payload_json TEXT,
                        lineage_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS fact_cards (
                        run_id TEXT NOT NULL,
                        fact_id TEXT NOT NULL,
                        requirement_id TEXT,
                        source_id TEXT NOT NULL,
                        fact TEXT NOT NULL,
                        metric TEXT,
                        value TEXT,
                        unit TEXT,
                        period TEXT,
                        scope TEXT,
                        allowed_use TEXT,
                        analysis_eligible INTEGER NOT NULL DEFAULT 0,
                        analysis_role TEXT,
                        source_level TEXT,
                        status TEXT NOT NULL,
                        payload_json TEXT,
                        input_hash TEXT,
                        output_hash TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, fact_id)
                    );

                    CREATE TABLE IF NOT EXISTS claim_units (
                        run_id TEXT NOT NULL,
                        claim_id TEXT NOT NULL,
                        payload_json TEXT,
                        requirement_ids_json TEXT,
                        fact_ids_json TEXT,
                        source_ids_json TEXT,
                        claim_strength TEXT,
                        claim_strength_ceiling TEXT,
                        status TEXT NOT NULL,
                        input_hash TEXT,
                        output_hash TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, claim_id)
                    );

                    CREATE TABLE IF NOT EXISTS sections (
                        run_id TEXT NOT NULL,
                        section_id TEXT NOT NULL,
                        payload_json TEXT,
                        requirement_ids_json TEXT,
                        claim_ids_json TEXT,
                        used_fact_refs_json TEXT,
                        evidence_backed INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL,
                        input_hash TEXT,
                        output_hash TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, section_id)
                    );

                    CREATE TABLE IF NOT EXISTS score_gaps (
                        run_id TEXT NOT NULL,
                        gap_id TEXT NOT NULL,
                        requirement_id TEXT,
                        chapter_id TEXT,
                        section_id TEXT,
                        gap_type TEXT NOT NULL,
                        severity TEXT,
                        missing_json TEXT,
                        retry_plan_json TEXT,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, gap_id)
                    );

                    CREATE TABLE IF NOT EXISTS lineage_edges (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        from_type TEXT NOT NULL,
                        from_id TEXT NOT NULL,
                        to_type TEXT NOT NULL,
                        to_id TEXT NOT NULL,
                        relation TEXT NOT NULL DEFAULT 'related',
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.executescript(
                    """
                    CREATE INDEX IF NOT EXISTS idx_artifacts_run_stage ON artifacts(run_id, stage);
                    CREATE INDEX IF NOT EXISTS idx_artifacts_requirement ON artifacts(requirement_id);
                    CREATE INDEX IF NOT EXISTS idx_sources_url_hash ON sources(canonical_url, content_hash);
                    CREATE INDEX IF NOT EXISTS idx_run_sources_canonical ON run_sources(canonical_source_id);
                    CREATE INDEX IF NOT EXISTS idx_fact_cards_requirement ON fact_cards(run_id, requirement_id, status);
                    CREATE INDEX IF NOT EXISTS idx_fact_cards_source ON fact_cards(run_id, source_id);
                    CREATE INDEX IF NOT EXISTS idx_claim_units_run ON claim_units(run_id, status);
                    CREATE INDEX IF NOT EXISTS idx_sections_run ON sections(run_id, status);
                    CREATE INDEX IF NOT EXISTS idx_score_gaps_requirement ON score_gaps(run_id, requirement_id, status);
                    CREATE INDEX IF NOT EXISTS idx_lineage_from ON lineage_edges(run_id, from_type, from_id);
                    CREATE INDEX IF NOT EXISTS idx_lineage_to ON lineage_edges(run_id, to_type, to_id);
                    """
                )
                conn.execute(f"PRAGMA user_version = {ARTIFACT_LEDGER_SCHEMA_VERSION}")
                conn.commit()
            self._initialized = True

    def _row(self, row: sqlite3.Row | None) -> Dict[str, Any]:
        return dict(row) if row is not None else {}

    def upsert_run(
        self,
        *,
        run_id: str,
        query: str,
        report_type: str = "",
        status: str = "running",
        user_id: str = "",
        freshness_policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled():
            return
        self._ensure_schema()
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, user_id, query, report_type, status, freshness_policy_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    query=excluded.query,
                    report_type=excluded.report_type,
                    status=excluded.status,
                    freshness_policy_json=excluded.freshness_policy_json,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    user_id,
                    query,
                    report_type,
                    status,
                    _json_dumps(freshness_policy or {}),
                    now,
                    now,
                ),
            )
            conn.commit()

    def upsert_evidence_requirement(
        self,
        *,
        run_id: str,
        requirement_id: str,
        chapter_id: str = "",
        hypothesis_id: str = "",
        proof_role: str,
        required_fields: Optional[Sequence[Any]] = None,
        min_source_level: Any = "",
        claim_strength_ceiling: str = "",
        freshness_required: bool = False,
        max_cache_age_hours: Optional[int] = None,
        status: str = "open",
        missing: Optional[Sequence[Any]] = None,
    ) -> None:
        self._ensure_schema()
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evidence_requirements(
                    run_id, requirement_id, chapter_id, hypothesis_id, proof_role, required_fields_json,
                    min_source_level, claim_strength_ceiling, freshness_required, max_cache_age_hours,
                    status, missing_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, requirement_id) DO UPDATE SET
                    chapter_id=excluded.chapter_id,
                    hypothesis_id=excluded.hypothesis_id,
                    proof_role=excluded.proof_role,
                    required_fields_json=excluded.required_fields_json,
                    min_source_level=excluded.min_source_level,
                    claim_strength_ceiling=excluded.claim_strength_ceiling,
                    freshness_required=excluded.freshness_required,
                    max_cache_age_hours=excluded.max_cache_age_hours,
                    status=excluded.status,
                    missing_json=excluded.missing_json,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    requirement_id,
                    chapter_id,
                    hypothesis_id,
                    proof_role,
                    _json_dumps(list(required_fields or [])),
                    _json_dumps(min_source_level) if isinstance(min_source_level, (list, tuple, set)) else str(min_source_level or ""),
                    claim_strength_ceiling,
                    1 if freshness_required else 0,
                    max_cache_age_hours,
                    status,
                    _json_dumps(list(missing or [])),
                    now,
                    now,
                ),
            )
            conn.commit()

    def get_evidence_requirement(self, run_id: str, requirement_id: str) -> Dict[str, Any]:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evidence_requirements WHERE run_id=? AND requirement_id=?",
                (run_id, requirement_id),
            ).fetchone()
        result = self._row(row)
        if result:
            result["required_fields"] = _decode_json(result.pop("required_fields_json", ""), [])
            result["missing"] = _decode_json(result.pop("missing_json", ""), [])
        return result

    def list_evidence_requirements(self, run_id: str, *, chapter_id: str = "", requirement_id: str = "") -> List[Dict[str, Any]]:
        self._ensure_schema()
        clauses = ["run_id=?"]
        params: List[Any] = [run_id]
        if chapter_id:
            clauses.append("chapter_id=?")
            params.append(chapter_id)
        if requirement_id:
            clauses.append("requirement_id=?")
            params.append(requirement_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM evidence_requirements WHERE {' AND '.join(clauses)} ORDER BY requirement_id",
                params,
            ).fetchall()
        return [self.get_evidence_requirement(row["run_id"], row["requirement_id"]) for row in rows]

    def _canonical_source_id(self, source: Dict[str, Any]) -> str:
        explicit = str(source.get("canonical_source_id") or "").strip()
        if explicit:
            return explicit
        identity = {
            "canonical_url": source.get("canonical_url") or source.get("url") or source.get("source_url") or "",
            "content_hash": source.get("content_hash") or "",
            "published_at": source.get("published_at") or source.get("date") or "",
            "title": source.get("title") or source.get("source_title") or "",
        }
        return "CS-" + _hash_payload(identity)[:20]

    def upsert_source(self, *, run_id: str, run_source_id: str, source: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_schema()
        payload = dict(as_dict(source))
        canonical_source_id = self._canonical_source_id(payload)
        run_ref = str(run_source_id or payload.get("source_id") or payload.get("ref") or payload.get("id") or canonical_source_id).strip()
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sources(
                    canonical_source_id, canonical_url, title, publisher, published_at, fetched_at,
                    source_type, source_level, verification_status, content_hash, storage_uri, status,
                    payload_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_source_id) DO UPDATE SET
                    canonical_url=excluded.canonical_url,
                    title=excluded.title,
                    publisher=excluded.publisher,
                    published_at=excluded.published_at,
                    fetched_at=excluded.fetched_at,
                    source_type=excluded.source_type,
                    source_level=excluded.source_level,
                    verification_status=excluded.verification_status,
                    content_hash=excluded.content_hash,
                    storage_uri=excluded.storage_uri,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    canonical_source_id,
                    payload.get("canonical_url") or payload.get("url") or payload.get("source_url") or "",
                    payload.get("title") or payload.get("source_title") or "",
                    payload.get("publisher") or payload.get("source") or "",
                    payload.get("published_at") or payload.get("date") or "",
                    payload.get("fetched_at") or "",
                    payload.get("source_type") or payload.get("type") or "",
                    str(payload.get("source_level") or payload.get("credibility") or "").strip().upper(),
                    payload.get("verification_status") or payload.get("source_verification_status") or "",
                    payload.get("content_hash") or "",
                    payload.get("storage_uri") or "",
                    payload.get("status") or "validated",
                    _json_dumps(payload),
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO run_sources(run_id, run_source_id, canonical_source_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, run_source_id) DO UPDATE SET
                    canonical_source_id=excluded.canonical_source_id,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (run_id, run_ref, canonical_source_id, payload.get("status") or "validated", now, now),
            )
            conn.commit()
        return {"canonical_source_id": canonical_source_id, "run_source_id": run_ref}

    def resolve_run_source(self, run_id: str, run_source_id: str) -> Dict[str, Any]:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT rs.run_source_id, rs.status AS run_source_status, s.*
                FROM run_sources rs
                JOIN sources s ON s.canonical_source_id = rs.canonical_source_id
                WHERE rs.run_id=? AND rs.run_source_id=?
                """,
                (run_id, run_source_id),
            ).fetchone()
        result = self._row(row)
        if result:
            result["payload"] = _decode_json(result.get("payload_json"), {})
        return result

    def _write_object(self, *, run_id: str, stage: str, artifact_id: str, payload: Any) -> Dict[str, Any]:
        path = self.object_root / "runs" / safe_path_part(run_id) / safe_path_part(stage) / f"{safe_path_part(artifact_id)}.json"
        data = _json_pretty(payload).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"storage_uri": str(path), "bytes": len(data)}

    def record_artifact(
        self,
        *,
        run_id: str,
        stage: str,
        artifact_type: str,
        payload: Any,
        status: str = "recorded",
        requirement_id: str = "",
        source_id: str = "",
        fact_id: str = "",
        claim_id: str = "",
        section_id: str = "",
        schema_version: str = "",
        prompt_version: str = "",
        model: str = "",
        producer_version: str = "",
        input_hash: str = "",
        output_hash: str = "",
        storage_uri: str = "",
        lineage: Optional[Dict[str, Any]] = None,
    ) -> ArtifactWriteResult:
        self._ensure_schema()
        encoded = _json_dumps(payload).encode("utf-8")
        output = output_hash or _hash_payload(payload)
        artifact_id = "ART-" + _hash_payload(
            {
                "run_id": run_id,
                "stage": stage,
                "artifact_type": artifact_type,
                "requirement_id": requirement_id,
                "source_id": source_id,
                "fact_id": fact_id,
                "claim_id": claim_id,
                "section_id": section_id,
                "input_hash": input_hash,
                "output_hash": output,
            }
        )[:24]
        payload_json = ""
        stored_uri = storage_uri
        storage_bytes = 0
        inline = False
        if storage_uri:
            storage_bytes = len(encoded)
        elif len(encoded) <= self.inline_max_bytes:
            payload_json = _json_dumps(payload)
            inline = True
        else:
            object_info = self._write_object(run_id=run_id, stage=stage, artifact_id=artifact_id, payload=payload)
            stored_uri = object_info["storage_uri"]
            storage_bytes = int(object_info["bytes"])
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, run_id, stage, artifact_type, status, requirement_id, source_id, fact_id,
                    claim_id, section_id, schema_version, prompt_version, model, producer_version,
                    input_hash, output_hash, content_hash, storage_uri, storage_bytes, payload_json,
                    lineage_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    status=excluded.status,
                    storage_uri=excluded.storage_uri,
                    storage_bytes=excluded.storage_bytes,
                    payload_json=excluded.payload_json,
                    lineage_json=excluded.lineage_json,
                    updated_at=excluded.updated_at
                """,
                (
                    artifact_id,
                    run_id,
                    stage,
                    artifact_type,
                    status,
                    requirement_id,
                    source_id,
                    fact_id,
                    claim_id,
                    section_id,
                    schema_version,
                    prompt_version,
                    model,
                    producer_version,
                    input_hash,
                    output,
                    output,
                    stored_uri,
                    storage_bytes,
                    payload_json,
                    _json_dumps(lineage or {}),
                    now,
                    now,
                ),
            )
            conn.commit()
        return ArtifactWriteResult(
            artifact_id=artifact_id,
            run_id=run_id,
            stage=stage,
            artifact_type=artifact_type,
            status=status,
            payload_inline=inline,
            storage_uri=stored_uri,
            output_hash=output,
            bytes=len(encoded),
        )

    def get_artifact(self, artifact_id: str) -> Dict[str, Any]:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        return self._row(row)

    def upsert_fact_card(
        self,
        *,
        run_id: str,
        fact_id: str,
        requirement_id: str = "",
        source_id: str,
        fact: str,
        metric: str = "",
        value: str = "",
        unit: str = "",
        period: str = "",
        scope: str = "",
        allowed_use: str = "",
        analysis_eligible: bool = False,
        analysis_role: str = "",
        source_level: str = "",
        status: str = "validated",
        payload: Optional[Dict[str, Any]] = None,
        input_hash: str = "",
        output_hash: str = "",
    ) -> None:
        self._ensure_schema()
        payload_json = _json_dumps(payload or {})
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fact_cards(
                    run_id, fact_id, requirement_id, source_id, fact, metric, value, unit, period, scope,
                    allowed_use, analysis_eligible, analysis_role, source_level, status, payload_json,
                    input_hash, output_hash, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, fact_id) DO UPDATE SET
                    requirement_id=excluded.requirement_id,
                    source_id=excluded.source_id,
                    fact=excluded.fact,
                    metric=excluded.metric,
                    value=excluded.value,
                    unit=excluded.unit,
                    period=excluded.period,
                    scope=excluded.scope,
                    allowed_use=excluded.allowed_use,
                    analysis_eligible=excluded.analysis_eligible,
                    analysis_role=excluded.analysis_role,
                    source_level=excluded.source_level,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    input_hash=excluded.input_hash,
                    output_hash=excluded.output_hash,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    fact_id,
                    requirement_id,
                    source_id,
                    fact,
                    metric,
                    value,
                    unit,
                    period,
                    scope,
                    allowed_use,
                    1 if analysis_eligible else 0,
                    analysis_role,
                    source_level,
                    status,
                    payload_json,
                    input_hash,
                    output_hash,
                    now,
                    now,
                ),
            )
            conn.commit()

    def list_fact_cards(
        self,
        run_id: str,
        *,
        requirement_id: str = "",
        fact_ids: Optional[Sequence[str]] = None,
        statuses: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_schema()
        clauses = ["run_id=?"]
        params: List[Any] = [run_id]
        if requirement_id:
            clauses.append("requirement_id=?")
            params.append(requirement_id)
        if fact_ids:
            placeholders = ",".join("?" for _ in fact_ids)
            clauses.append(f"fact_id IN ({placeholders})")
            params.extend(list(fact_ids))
        if statuses:
            status_list = list(statuses)
            placeholders = ",".join("?" for _ in status_list)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status_list)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM fact_cards WHERE {' AND '.join(clauses)} ORDER BY fact_id",
                params,
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["analysis_eligible"] = bool(item.get("analysis_eligible"))
            item["payload"] = _decode_json(item.get("payload_json"), {})
            result.append(item)
        return result

    def upsert_claim_unit(
        self,
        *,
        run_id: str,
        claim_id: str,
        payload: Dict[str, Any],
        requirement_ids: Optional[Sequence[Any]] = None,
        fact_ids: Optional[Sequence[Any]] = None,
        source_ids: Optional[Sequence[Any]] = None,
        status: str = "validated",
        input_hash: str = "",
        output_hash: str = "",
    ) -> None:
        self._ensure_schema()
        reqs = [str(item) for item in (requirement_ids or as_list(payload.get("requirement_ids"))) if str(item).strip()]
        facts = [str(item) for item in (fact_ids or as_list(payload.get("fact_ids")) or as_list(payload.get("used_fact_refs")) or as_list(payload.get("evidence_refs"))) if str(item).strip()]
        sources = [str(item) for item in (source_ids or as_list(payload.get("source_ids"))) if str(item).strip()]
        claim_text = str(payload.get("claim") or payload.get("claim_text") or "").strip()
        effective_status = _enforced_claim_status(
            status, facts=facts, sources=sources, claim_text=claim_text, requirement_ids=reqs
        )
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO claim_units(
                    run_id, claim_id, payload_json, requirement_ids_json, fact_ids_json, source_ids_json,
                    claim_strength, claim_strength_ceiling, status, input_hash, output_hash, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, claim_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    requirement_ids_json=excluded.requirement_ids_json,
                    fact_ids_json=excluded.fact_ids_json,
                    source_ids_json=excluded.source_ids_json,
                    claim_strength=excluded.claim_strength,
                    claim_strength_ceiling=excluded.claim_strength_ceiling,
                    status=excluded.status,
                    input_hash=excluded.input_hash,
                    output_hash=excluded.output_hash,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    claim_id,
                    _json_dumps(payload),
                    _json_dumps(reqs),
                    _json_dumps(facts),
                    _json_dumps(sources),
                    payload.get("claim_strength") or payload.get("strength") or "",
                    payload.get("claim_strength_ceiling") or "",
                    effective_status,
                    input_hash,
                    output_hash,
                    now,
                    now,
                ),
            )
            conn.commit()

    def list_claim_units(self, run_id: str, *, claim_ids: Optional[Sequence[str]] = None, requirement_id: str = "") -> List[Dict[str, Any]]:
        self._ensure_schema()
        clauses = ["run_id=?"]
        params: List[Any] = [run_id]
        if claim_ids:
            placeholders = ",".join("?" for _ in claim_ids)
            clauses.append(f"claim_id IN ({placeholders})")
            params.extend(list(claim_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM claim_units WHERE {' AND '.join(clauses)} ORDER BY claim_id",
                params,
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            payload = _decode_json(item.get("payload_json"), {})
            item.update(payload)
            item["payload"] = payload
            item["requirement_ids"] = _decode_json(item.get("requirement_ids_json"), [])
            item["fact_ids"] = _decode_json(item.get("fact_ids_json"), [])
            item["source_ids"] = _decode_json(item.get("source_ids_json"), [])
            if requirement_id and requirement_id not in item["requirement_ids"]:
                continue
            result.append(item)
        return result

    def upsert_section(
        self,
        *,
        run_id: str,
        section_id: str,
        payload: Dict[str, Any],
        requirement_ids: Optional[Sequence[Any]] = None,
        claim_ids: Optional[Sequence[Any]] = None,
        used_fact_refs: Optional[Sequence[Any]] = None,
        evidence_backed: bool = False,
        status: str = "validated",
        input_hash: str = "",
        output_hash: str = "",
    ) -> None:
        self._ensure_schema()
        reqs = [str(item) for item in (requirement_ids or as_list(payload.get("requirement_ids"))) if str(item).strip()]
        claims = [str(item) for item in (claim_ids or as_list(payload.get("claim_ids")) or ([payload.get("claim_id")] if payload.get("claim_id") else [])) if str(item).strip()]
        refs = [str(item) for item in (used_fact_refs or as_list(payload.get("used_fact_refs")) or as_list(payload.get("evidence_refs"))) if str(item).strip()]
        effective_status = _enforced_section_status(status, claim_ids=claims, used_fact_refs=refs)
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sections(
                    run_id, section_id, payload_json, requirement_ids_json, claim_ids_json, used_fact_refs_json,
                    evidence_backed, status, input_hash, output_hash, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, section_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    requirement_ids_json=excluded.requirement_ids_json,
                    claim_ids_json=excluded.claim_ids_json,
                    used_fact_refs_json=excluded.used_fact_refs_json,
                    evidence_backed=excluded.evidence_backed,
                    status=excluded.status,
                    input_hash=excluded.input_hash,
                    output_hash=excluded.output_hash,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    section_id,
                    _json_dumps(payload),
                    _json_dumps(reqs),
                    _json_dumps(claims),
                    _json_dumps(refs),
                    1 if evidence_backed else 0,
                    effective_status,
                    input_hash,
                    output_hash,
                    now,
                    now,
                ),
            )
            conn.commit()

    def get_section(self, run_id: str, section_id: str) -> Dict[str, Any]:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sections WHERE run_id=? AND section_id=?", (run_id, section_id)).fetchone()
        item = self._row(row)
        if not item:
            return {}
        payload = _decode_json(item.get("payload_json"), {})
        item.update(payload)
        item["payload"] = payload
        item["requirement_ids"] = _decode_json(item.get("requirement_ids_json"), [])
        item["claim_ids"] = _decode_json(item.get("claim_ids_json"), [])
        item["used_fact_refs"] = _decode_json(item.get("used_fact_refs_json"), [])
        item["evidence_backed"] = bool(item.get("evidence_backed"))
        return item

    def upsert_score_gap(
        self,
        *,
        run_id: str,
        gap_id: str,
        requirement_id: str = "",
        chapter_id: str = "",
        section_id: str = "",
        gap_type: str,
        severity: str = "",
        missing: Optional[Sequence[Any]] = None,
        retry_plan: Optional[Dict[str, Any]] = None,
        status: str = "open",
    ) -> None:
        self._ensure_schema()
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO score_gaps(
                    run_id, gap_id, requirement_id, chapter_id, section_id, gap_type, severity,
                    missing_json, retry_plan_json, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, gap_id) DO UPDATE SET
                    requirement_id=excluded.requirement_id,
                    chapter_id=excluded.chapter_id,
                    section_id=excluded.section_id,
                    gap_type=excluded.gap_type,
                    severity=excluded.severity,
                    missing_json=excluded.missing_json,
                    retry_plan_json=excluded.retry_plan_json,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    gap_id,
                    requirement_id,
                    chapter_id,
                    section_id,
                    gap_type,
                    severity,
                    _json_dumps(list(missing or [])),
                    _json_dumps(retry_plan or {}),
                    status,
                    now,
                    now,
                ),
            )
            conn.commit()

    def list_score_gaps(
        self,
        run_id: str,
        *,
        gap_id: str = "",
        requirement_id: str = "",
        statuses: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_schema()
        clauses = ["run_id=?"]
        params: List[Any] = [run_id]
        if gap_id:
            clauses.append("gap_id=?")
            params.append(gap_id)
        if requirement_id:
            clauses.append("requirement_id=?")
            params.append(requirement_id)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(list(statuses))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM score_gaps WHERE {' AND '.join(clauses)} ORDER BY gap_id",
                params,
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["missing"] = _decode_json(item.get("missing_json"), [])
            item["retry_plan"] = _decode_json(item.get("retry_plan_json"), {})
            result.append(item)
        return result

    def add_lineage_edge(
        self,
        run_id: str,
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
        relation: str = "related",
    ) -> bool:
        if not self.enabled():
            return False
        self._ensure_schema()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO lineage_edges(run_id, from_type, from_id, to_type, to_id, relation, created_at)
                SELECT ?, ?, ?, ?, ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM lineage_edges
                    WHERE run_id = ?
                      AND from_type = ?
                      AND from_id = ?
                      AND to_type = ?
                      AND to_id = ?
                      AND relation = ?
                )
                """,
                (
                    run_id,
                    from_type,
                    from_id,
                    to_type,
                    to_id,
                    relation,
                    _now_iso(),
                    run_id,
                    from_type,
                    from_id,
                    to_type,
                    to_id,
                    relation,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def count_lineage_edges(self, run_id: str) -> int:
        """Total persisted lineage edges for a run (not just edges added this call)."""
        if not self.enabled():
            return 0
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM lineage_edges WHERE run_id=?",
                (run_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def traverse_lineage(self, run_id: str, from_type: str, from_id: str, *, max_depth: int = 4) -> List[Dict[str, Any]]:
        self._ensure_schema()
        result: List[Dict[str, Any]] = []
        seen = {(from_type, from_id)}
        queue = deque([(from_type, from_id, 0)])
        with self._connect() as conn:
            while queue:
                current_type, current_id, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                rows = conn.execute(
                    """
                    SELECT run_id, from_type, from_id, to_type, to_id, relation, created_at
                    FROM lineage_edges
                    WHERE run_id=? AND from_type=? AND from_id=?
                    ORDER BY id
                    """,
                    (run_id, current_type, current_id),
                ).fetchall()
                for row in rows:
                    item = dict(row)
                    item["depth"] = depth + 1
                    result.append(item)
                    key = (item["to_type"], item["to_id"])
                    if key not in seen:
                        seen.add(key)
                        queue.append((item["to_type"], item["to_id"], depth + 1))
        return result

    def store_source_object(self, canonical_source_id: str, name: str, payload: Any) -> Dict[str, Any]:
        encoded = _json_pretty(payload).encode("utf-8")
        path = self.object_root / "sources" / safe_path_part(canonical_source_id) / safe_path_part(name)
        if path.suffix == "":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        return {"storage_uri": str(path), "content_hash": _hash_payload(payload), "bytes": len(encoded)}


def default_artifact_store() -> ArtifactStore:
    return ArtifactStore()
