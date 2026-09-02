"""Redis-backed request-level tool traces with defensive argument redaction."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


_SENSITIVE_PARTS = (
    "password",
    "passwd",
    "pwd",
    "token",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "card_number",
    "cvv",
)


def sanitize_trace_value(value: Any, *, depth: int = 0) -> Any:
    """Return a JSON-safe, size-bounded value suitable for diagnostic storage."""
    if depth >= 5:
        return "[TRUNCATED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + "...[TRUNCATED]"
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in list(value.items())[:50]:
            key_text = str(key)
            normalized = key_text.lower().replace("-", "_")
            if any(part in normalized for part in _SENSITIVE_PARTS):
                cleaned[key_text] = "[REDACTED]"
            else:
                cleaned[key_text] = sanitize_trace_value(item, depth=depth + 1)
        return cleaned
    if isinstance(value, (list, tuple, set)):
        return [sanitize_trace_value(item, depth=depth + 1) for item in list(value)[:50]]
    return sanitize_trace_value(str(value), depth=depth + 1)


@dataclass(frozen=True)
class ToolTraceRecord:
    request_id: str
    timestamp: float
    caller: str
    user_id: str
    conv_id: str
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    denied: bool
    cached: bool
    reranked: bool
    latency_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RedisToolTraceStore:
    """Store per-request traces and a bounded recent-trace feed in Redis."""

    def __init__(
        self,
        redis_url: str,
        *,
        ttl_s: int = 86400,
        recent_limit: int = 1000,
    ):
        import redis.asyncio as redis

        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._ttl_s = max(60, int(ttl_s))
        self._recent_limit = max(1, int(recent_limit))

    @staticmethod
    def _request_key(request_id: str) -> str:
        return f"pkkmind:tool_trace:request:{request_id}"

    @staticmethod
    def _recent_key() -> str:
        return "pkkmind:tool_trace:recent"

    async def record(
        self,
        *,
        request_id: str,
        caller: str,
        user_id: str,
        conv_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        success: bool,
        denied: bool,
        cached: bool,
        reranked: bool,
        latency_ms: float,
        error: Optional[str],
    ) -> ToolTraceRecord:
        record = ToolTraceRecord(
            request_id=request_id,
            timestamp=time.time(),
            caller=caller,
            user_id=str(sanitize_trace_value(user_id)),
            conv_id=str(sanitize_trace_value(conv_id)),
            tool_name=tool_name,
            arguments=sanitize_trace_value(arguments),
            success=success,
            denied=denied,
            cached=cached,
            reranked=reranked,
            latency_ms=round(float(latency_ms or 0.0), 3),
            error=sanitize_trace_value(error) if error else None,
        )
        payload = json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"))
        request_key = self._request_key(request_id)
        recent_key = self._recent_key()
        pipe = self._redis.pipeline(transaction=False)
        pipe.rpush(request_key, payload)
        pipe.expire(request_key, self._ttl_s)
        pipe.lpush(recent_key, payload)
        pipe.ltrim(recent_key, 0, self._recent_limit - 1)
        pipe.expire(recent_key, self._ttl_s)
        await pipe.execute()
        return record

    async def get_request(self, request_id: str) -> List[Dict[str, Any]]:
        values = await self._redis.lrange(self._request_key(request_id), 0, -1)
        return [json.loads(value) for value in values]

    async def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        bounded = max(1, min(int(limit), 100))
        values = await self._redis.lrange(self._recent_key(), 0, bounded - 1)
        return [json.loads(value) for value in values]

    async def close(self) -> None:
        await self._redis.aclose()
