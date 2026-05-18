from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


def json_safe_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}
    if isinstance(value, (datetime, date)):
        return {"__type__": type(value).__name__, "value": value.isoformat()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set):
        return sorted(value, key=lambda item: repr(item))
    if isinstance(value, bytes):
        return {"__type__": "bytes", "hex": value.hex()}
    if hasattr(value, "tolist"):
        return value.tolist()
    return {"__type__": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def make_cache_key(namespace: str, payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_safe_default)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


class TTLCache:
    def __init__(self, *, ttl_seconds: int, max_items: int):
        self.ttl_seconds = max(0, int(ttl_seconds or 0))
        self.max_items = max(0, int(max_items or 0))
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()

    def enabled(self) -> bool:
        return self.ttl_seconds > 0 and self.max_items > 0

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled():
            return None
        now = time.time()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return copy.deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        if not self.enabled():
            return
        expires_at = time.time() + self.ttl_seconds
        with self._lock:
            self._items[key] = (expires_at, copy.deepcopy(value))
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def stats(self) -> Dict[str, int]:
        now = time.time()
        with self._lock:
            expired = [key for key, (expires_at, _) in self._items.items() if expires_at <= now]
            for key in expired:
                self._items.pop(key, None)
            return {
                "enabled": 1 if self.enabled() else 0,
                "items": len(self._items),
                "ttl_seconds": self.ttl_seconds,
                "max_items": self.max_items,
            }
