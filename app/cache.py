"""SHA-256 content cache.

Keyed on the file bytes + the expected doc-type + the model version, so a
re-upload of an identical file (very common — clients resend the same scan)
costs nothing. Uses Redis when REDIS_URL is set, otherwise an in-process LRU
that's perfectly fine for a single instance.
"""
import hashlib
import json
import time
from collections import OrderedDict
from typing import Optional

from .config import get_settings

try:  # redis is optional at runtime
    import redis as _redis
except Exception:  # pragma: no cover
    _redis = None


def content_hash(data: bytes, *parts: str) -> str:
    h = hashlib.sha256()
    h.update(data)
    for p in parts:
        h.update(b"\x00")
        h.update((p or "").encode())
    return "docai:" + h.hexdigest()


class _MemoryLRU:
    def __init__(self, maxitems: int):
        self.maxitems = maxitems
        self.store: "OrderedDict[str, tuple[float, str]]" = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        v = self.store.get(key)
        if not v:
            return None
        exp, payload = v
        if exp and exp < time.time():
            self.store.pop(key, None)
            return None
        self.store.move_to_end(key)
        return payload

    def set(self, key: str, payload: str, ttl: int) -> None:
        self.store[key] = (time.time() + ttl if ttl else 0, payload)
        self.store.move_to_end(key)
        while len(self.store) > self.maxitems:
            self.store.popitem(last=False)


class Cache:
    def __init__(self) -> None:
        s = get_settings()
        self.ttl = s.cache_ttl_seconds
        self.client = None
        if s.redis_url and _redis is not None:
            try:
                self.client = _redis.from_url(s.redis_url, decode_responses=True)
                self.client.ping()
            except Exception:
                self.client = None
        self.mem = _MemoryLRU(s.cache_max_items)

    def get(self, key: str) -> Optional[dict]:
        raw = None
        if self.client:
            try:
                raw = self.client.get(key)
            except Exception:
                raw = None
        if raw is None:
            raw = self.mem.get(key)
        return json.loads(raw) if raw else None

    def set(self, key: str, value: dict) -> None:
        raw = json.dumps(value)
        if self.client:
            try:
                self.client.set(key, raw, ex=self.ttl)
            except Exception:
                pass
        self.mem.set(key, raw, self.ttl)


_cache: Optional[Cache] = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache
