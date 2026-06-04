from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


ARTIFACT_LEDGER_SCHEMA_VERSION = 1

USABLE_FACT_STATUSES = {"validated", "admissible"}
UNUSABLE_FACT_STATUSES = {"rejected", "stale", "superseded", "excluded", "appendix_only"}


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    run_id: str
    stage: str
    artifact_type: str
    status: str
    storage_uri: str = ""
    output_hash: str = ""


@dataclass(frozen=True)
class LineageEdge:
    run_id: str
    from_type: str
    from_id: str
    to_type: str
    to_id: str
    relation: str = "related"


@dataclass(frozen=True)
class ArtifactWriteResult:
    artifact_id: str
    run_id: str
    stage: str
    artifact_type: str
    status: str
    payload_inline: bool
    storage_uri: str = ""
    output_hash: str = ""
    bytes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key in self.extra:
            return self.extra[key]
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.extra:
            return self.extra[key]
        return getattr(self, key, default)


def is_usable_fact_status(status: str) -> bool:
    return str(status or "").strip().lower() in USABLE_FACT_STATUSES


def as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
