from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, TypeVar

JsonObject = dict[str, Any]
T = TypeVar("T", bound=JsonObject)


class ResponseCache(Protocol):
    """Contract used by API services for optional shared response caching."""

    def get_json(self, key: str) -> JsonObject | None: ...

    def set_json(self, key: str, value: Mapping[str, Any], ttl_seconds: int) -> None: ...

    def health(self) -> JsonObject: ...

    def close(self) -> None: ...


class NullResponseCache:
    """No-op cache used when Redis is disabled or unavailable at startup."""

    def __init__(self, configured: bool = False, reason: str | None = None) -> None:
        self.configured = configured
        self.reason = reason

    def get_json(self, key: str) -> JsonObject | None:
        return None

    def set_json(self, key: str, value: Mapping[str, Any], ttl_seconds: int) -> None:
        return None

    def health(self) -> JsonObject:
        status = "unavailable" if self.configured else "disabled"
        return {
            "backend": "redis" if self.configured else "none",
            "status": status,
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "reason": self.reason,
        }

    def close(self) -> None:
        return None


class RedisResponseCache:
    """Redis-backed JSON cache that fails open when the cache becomes unavailable."""

    def __init__(self, client: Any, failure_cooldown_seconds: float = 30.0) -> None:
        self.client = client
        self.failure_cooldown_seconds = failure_cooldown_seconds
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.bypasses = 0
        self.last_error: str | None = None
        self._disabled_until = 0.0

    def _available(self) -> bool:
        return time.monotonic() >= self._disabled_until

    def _record_success(self) -> None:
        self.last_error = None
        self._disabled_until = 0.0

    def _record_failure(self, error: Exception) -> None:
        self.errors += 1
        self.last_error = f"{type(error).__name__}: {error}"
        self._disabled_until = time.monotonic() + self.failure_cooldown_seconds

    def get_json(self, key: str) -> JsonObject | None:
        if not self._available():
            self.bypasses += 1
            return None
        try:
            cached = self.client.get(key)
            self._record_success()
            if cached is None:
                self.misses += 1
                return None
            value = json.loads(cached)
            if not isinstance(value, dict):
                raise TypeError("Cached API response must be a JSON object.")
            self.hits += 1
            return value
        except Exception as error:  # noqa: BLE001 - Cache failures must not fail API requests.
            self._record_failure(error)
            return None

    def set_json(self, key: str, value: Mapping[str, Any], ttl_seconds: int) -> None:
        if not self._available() or ttl_seconds <= 0:
            return
        try:
            payload = json.dumps(value, allow_nan=False, separators=(",", ":"))
            self.client.setex(key, ttl_seconds, payload)
            self._record_success()
        except Exception as error:  # noqa: BLE001 - Cache failures must not fail API requests.
            self._record_failure(error)

    def health(self) -> JsonObject:
        requests = self.hits + self.misses
        return {
            "backend": "redis",
            "status": "ready" if self._available() and self.last_error is None else "degraded",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / requests if requests else 0.0,
            "errors": self.errors,
            "bypasses": self.bypasses,
            "last_error": self.last_error,
        }

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001 - Shutdown remains best-effort.
            return


def build_runtime_version(data_dir: str | Path, artifact_dir: str | Path) -> str:
    """Return a stable cache namespace derived from serving data and artifacts."""
    explicit_version = os.getenv("DATA_VERSION")
    if explicit_version:
        return explicit_version

    digest = hashlib.sha256()
    files: list[Path] = []
    for root in (Path(data_dir).expanduser().resolve(), Path(artifact_dir).expanduser().resolve()):
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    for path in sorted(files):
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()[:16]


def build_cache_key(namespace: str, runtime_version: str, payload: Mapping[str, Any]) -> str:
    """Build a deterministic Redis key without embedding the full request payload."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    request_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
    return f"nba-scout:{runtime_version}:{namespace}:{request_hash}"


def get_or_set_cached_json(
    cache: ResponseCache,
    key: str,
    ttl_seconds: int,
    builder: Callable[[], T],
) -> T:
    """Return a cached response or compute and cache one successful response."""
    cached = cache.get_json(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    value = builder()
    cache.set_json(key, value, ttl_seconds)
    return value


def build_response_cache() -> ResponseCache:
    """Create the configured Redis cache or a fail-open no-op cache."""
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return NullResponseCache(reason="REDIS_URL is not configured")

    connect_timeout = float(os.getenv("REDIS_CONNECT_TIMEOUT_SECONDS", "0.5"))
    socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "0.5"))
    cooldown = float(os.getenv("REDIS_FAILURE_COOLDOWN_SECONDS", "30"))
    try:
        from redis import Redis

        client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=connect_timeout,
            socket_timeout=socket_timeout,
            health_check_interval=30,
        )
        client.ping()
        return RedisResponseCache(client, failure_cooldown_seconds=cooldown)
    except Exception as error:  # noqa: BLE001 - A missing cache must not block API startup.
        return NullResponseCache(
            configured=True,
            reason=f"{type(error).__name__}: {error}",
        )


def cache_ttl_seconds(name: str, default: int) -> int:
    """Read one non-negative cache TTL from the environment."""
    return max(0, int(os.getenv(name, str(default))))
