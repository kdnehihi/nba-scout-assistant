from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

import app.main as api_main
from app.services.cache import (
    NullResponseCache,
    RedisResponseCache,
    build_cache_key,
    get_or_set_cached_json,
)
from tests.test_api_routes import make_resources


class FakeRedis:
    """Minimal Redis client used to verify response-cache behavior."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.closed = False

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        assert ttl_seconds > 0
        self.values[key] = value

    def close(self) -> None:
        self.closed = True


class FailingRedis(FakeRedis):
    def get(self, key: str) -> str | None:
        raise ConnectionError("Redis unavailable")


def test_cache_key_is_stable_and_namespaced_by_runtime_version():
    payload = {"player_name": "LeBron James", "season": "2024-25"}
    first = build_cache_key("recommendations", "data-v1", payload)
    second = build_cache_key("recommendations", "data-v1", dict(reversed(payload.items())))
    changed = build_cache_key("recommendations", "data-v2", payload)

    assert first == second
    assert first != changed
    assert "LeBron" not in first


def test_redis_cache_reuses_successful_json_response():
    cache = RedisResponseCache(FakeRedis())
    calls = 0

    def build() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": 42}

    assert get_or_set_cached_json(cache, "key", 60, build) == {"value": 42}
    assert get_or_set_cached_json(cache, "key", 60, build) == {"value": 42}
    assert calls == 1
    assert cache.health()["hits"] == 1
    assert cache.health()["misses"] == 1


def test_redis_failure_opens_circuit_and_falls_back_to_builder():
    cache = RedisResponseCache(FailingRedis(), failure_cooldown_seconds=60)
    calls = 0

    def build() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"status": "computed"}

    assert get_or_set_cached_json(cache, "key", 60, build) == {"status": "computed"}
    assert get_or_set_cached_json(cache, "key", 60, build) == {"status": "computed"}
    assert calls == 2
    assert cache.health()["status"] == "degraded"
    assert cache.health()["errors"] == 1
    assert cache.health()["bypasses"] == 1


def test_recommendation_endpoint_uses_shared_response_cache(monkeypatch):
    cache = RedisResponseCache(FakeRedis())
    resources = replace(
        make_resources(with_ranker=False),
        response_cache=cache,
        runtime_version="api-test-v1",
    )
    monkeypatch.setattr(api_main, "load_app_resources", lambda **_: resources)
    payload = {
        "player_name": "Target Guard",
        "season": "2024-25",
        "top_n": 2,
        "preset": "playing_profile",
    }

    with TestClient(api_main.app) as client:
        first = client.post("/recommendations", json=payload)
        second = client.post("/recommendations", json=payload)
        health = client.get("/health")

    assert first.status_code == 200
    assert second.json() == first.json()
    assert health.json()["cache"]["hits"] == 1
    assert health.json()["cache"]["misses"] == 1


def test_null_cache_reports_disabled_without_affecting_responses():
    cache = NullResponseCache(reason="not configured")
    assert cache.get_json("anything") is None
    assert cache.health()["status"] == "disabled"
