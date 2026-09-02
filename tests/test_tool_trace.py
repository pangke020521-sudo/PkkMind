import asyncio
import unittest

from mcp.tool_manager import MCPToolManager, Tool, ToolCallContext
from mcp.tool_trace import sanitize_trace_value


class _FakeClient:
    def __init__(self):
        self.messages = self

    async def create(self, **kwargs):
        raise AssertionError("LLM should not be called")


class _TraceStore:
    def __init__(self):
        self.records = []

    async def record(self, **kwargs):
        kwargs["arguments"] = sanitize_trace_value(kwargs["arguments"])
        self.records.append(kwargs)

    async def get_request(self, request_id):
        return [item for item in self.records if item["request_id"] == request_id]

    async def get_recent(self, limit):
        return list(reversed(self.records))[:limit]

    async def close(self):
        return None


class ToolTraceTests(unittest.TestCase):
    def test_sensitive_arguments_are_redacted(self):
        value = sanitize_trace_value({
            "query": "hello",
            "api_key": "secret-value",
            "nested": {"Authorization": "Bearer token", "safe": 1},
        })

        self.assertEqual(value["query"], "hello")
        self.assertEqual(value["api_key"], "[REDACTED]")
        self.assertEqual(value["nested"]["Authorization"], "[REDACTED]")
        self.assertEqual(value["nested"]["safe"], 1)

    def test_manager_records_success_and_denial_by_request_id(self):
        trace_store = _TraceStore()
        manager = MCPToolManager(
            _FakeClient(),
            model="test",
            trace_store=trace_store,
        )

        async def handler(params, context):
            return {"ok": True}

        manager.register(Tool(
            name="technical_only",
            description="test",
            handler=handler,
            schema={"type": "object", "properties": {}},
            allowed_callers=("technical",),
        ))
        allowed_context = ToolCallContext("req-1", "technical", "u1", "c1")
        denied_context = ToolCallContext("req-2", "billing", "u2", "c2")

        asyncio.run(manager.call(
            "technical_only",
            {"token": "hidden"},
            allowed_context,
        ))
        asyncio.run(manager.call(
            "technical_only",
            {},
            denied_context,
        ))

        allowed = asyncio.run(manager.get_request_trace("req-1"))
        denied = asyncio.run(manager.get_request_trace("req-2"))
        self.assertTrue(allowed[0]["success"])
        self.assertEqual(allowed[0]["arguments"]["token"], "[REDACTED]")
        self.assertTrue(denied[0]["denied"])


if __name__ == "__main__":
    unittest.main()
