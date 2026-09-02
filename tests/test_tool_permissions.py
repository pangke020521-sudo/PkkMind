import asyncio
import json
import unittest
from types import SimpleNamespace

from agents.agent_orchestrator import AgentOrchestrator, AgentType, Request
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from mcp.domain_tools import create_domain_tools
from mcp.tool_manager import MCPToolManager, Tool, ToolCallContext


class _FakeClient:
    def __init__(self, text="ok"):
        self.text = text
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)]
        )


def _tool_context(caller: str) -> ToolCallContext:
    return ToolCallContext(
        request_id="req-test",
        caller=caller,
        user_id="u1",
        conv_id="c1",
    )


def _request(intent: IntentCategory, intent_group: str, entities=None) -> Request:
    return Request(
        message="请协助处理",
        user_id="u1",
        conv_id="c1",
        intent=intent,
        intent_group=intent_group,
        urgency=UrgencyLevel.MEDIUM,
        entities=entities or {},
        request_id="req-test",
    )


class ToolPermissionTests(unittest.TestCase):
    def test_authorized_agent_can_execute_tool(self):
        calls = []

        async def handler(params, context):
            calls.append((params, context))
            return {"ok": True}

        manager = MCPToolManager(_FakeClient(), model="test")
        manager.register(Tool(
            name="technical_only",
            description="test",
            handler=handler,
            schema={"type": "object", "properties": {}},
            allowed_callers=("technical",),
        ))

        result = asyncio.run(manager.call(
            "technical_only",
            {},
            _tool_context("technical"),
        ))

        self.assertTrue(result.success)
        self.assertFalse(result.denied)
        self.assertEqual(len(calls), 1)

    def test_unauthorized_agent_is_denied_before_handler_and_fallback(self):
        handler_calls = []
        fallback_calls = []

        async def handler(params, context):
            handler_calls.append((params, context))
            return {"ok": True}

        def fallback(params, context, error):
            fallback_calls.append((params, context, error))
            return {"fallback": True}

        manager = MCPToolManager(_FakeClient(), model="test")
        manager.register(Tool(
            name="technical_only",
            description="test",
            handler=handler,
            schema={"type": "object", "properties": {}},
            fallback=fallback,
            allowed_callers=("technical",),
        ))

        result = asyncio.run(manager.call(
            "technical_only",
            {},
            _tool_context("billing"),
        ))

        self.assertFalse(result.success)
        self.assertTrue(result.denied)
        self.assertIn("无权", result.error)
        self.assertEqual(handler_calls, [])
        self.assertEqual(fallback_calls, [])
        self.assertEqual(manager.get_stats()["technical_only"]["denied"], 1)

    def test_restricted_search_is_denied_before_query_rewrite(self):
        client = _FakeClient("不应调用模型")
        manager = MCPToolManager(client, model="test")
        manager.register(Tool(
            name="knowledge_search",
            description="test",
            handler=lambda params, context: [],
            schema={"type": "object", "properties": {"query": {"type": "string"}}},
            allowed_callers=("system",),
        ))

        result = asyncio.run(manager.search_with_rewrite(
            "knowledge_search",
            "退款流程",
            _tool_context("escalation"),
        ))

        self.assertTrue(result.denied)
        self.assertEqual(client.calls, [])


class DomainToolTests(unittest.TestCase):
    def setUp(self):
        self.manager = MCPToolManager(_FakeClient(), model="test")
        for tool in create_domain_tools():
            self.manager.register(tool)

    def test_error_code_lookup_is_technical_only(self):
        allowed = asyncio.run(self.manager.call(
            "error_code_lookup",
            {"error_code": "401"},
            _tool_context("technical"),
        ))
        denied = asyncio.run(self.manager.call(
            "error_code_lookup",
            {"error_code": "401"},
            _tool_context("billing"),
        ))

        self.assertEqual(allowed.data["meaning"], "身份认证失败或凭证已失效")
        self.assertTrue(denied.denied)

    def test_billing_field_check_reports_missing_fields_without_side_effects(self):
        result = asyncio.run(self.manager.call(
            "billing_field_check",
            {"entities": {"order_id": ["ORD-1"], "amount": [], "date": []}},
            _tool_context("billing"),
        ))

        self.assertTrue(result.success)
        self.assertEqual(result.data["available_fields"], ["order_id"])
        self.assertEqual(result.data["missing_fields"], ["amount", "date"])
        self.assertFalse(result.data["side_effects_performed"])

    def test_error_code_entity_supports_chinese_labels_without_matching_order_id(self):
        recognizer = IntentRecognizer(_FakeClient(), embedding_enabled=False)

        entities = recognizer._extract_entities("订单ORD-TEST-2登录时显示错误码401")

        self.assertEqual(entities["order_id"], ["ORD-TEST-2"])
        self.assertEqual(entities["error_code"], ["401"])


class AgentToolContextTests(unittest.TestCase):
    def _orchestrator(self):
        client = _FakeClient("处理完成")
        manager = MCPToolManager(client, model="test")
        for tool in create_domain_tools():
            manager.register(tool)
        orchestrator = AgentOrchestrator(
            client=client,
            model="test-model",
            embedding_enabled=False,
            provider="openai",
            tool_manager=manager,
        )
        return orchestrator, client

    def test_technical_agent_receives_error_tool_result(self):
        orchestrator, client = self._orchestrator()
        req = _request(
            IntentCategory.TECHNICAL_LOGIN,
            "technical",
            {"error_code": ["401"]},
        )

        result = asyncio.run(orchestrator.run(req))

        self.assertEqual(result.agent_type.value, "technical")
        model_call = client.calls[-1]
        self.assertIn("error_code_lookup", json.dumps(model_call["messages"], ensure_ascii=False))
        self.assertIn("身份认证失败", json.dumps(model_call["messages"], ensure_ascii=False))

    def test_billing_agent_does_not_receive_technical_tool_result(self):
        orchestrator, client = self._orchestrator()
        req = _request(
            IntentCategory.PAYMENT_ISSUE,
            "billing",
            {"order_id": ["ORD-1"], "amount": ["99元"], "date": []},
        )

        result = asyncio.run(orchestrator.run(req))

        self.assertEqual(result.agent_type.value, "billing")
        model_messages = json.dumps(client.calls[-1]["messages"], ensure_ascii=False)
        self.assertIn("billing_field_check", model_messages)
        self.assertNotIn("error_code_lookup", model_messages)

    def test_parallel_agents_receive_isolated_tool_contexts(self):
        orchestrator, client = self._orchestrator()
        req = _request(
            IntentCategory.PAYMENT_ISSUE,
            "billing",
            {
                "error_code": ["401"],
                "order_id": ["ORD-1"],
                "amount": ["99元"],
                "date": [],
            },
        )

        async def execute_both():
            return await asyncio.gather(
                orchestrator._execute(req, AgentType.TECHNICAL),
                orchestrator._execute(req, AgentType.BILLING),
            )

        asyncio.run(execute_both())

        calls_by_role = {
            call["system"]: json.dumps(call["messages"], ensure_ascii=False)
            for call in client.calls
        }
        technical_messages = next(
            messages for system, messages in calls_by_role.items()
            if "技术故障诊断与排障" in system
        )
        billing_messages = next(
            messages for system, messages in calls_by_role.items()
            if "账单核验与售后处理" in system
        )
        self.assertIn("error_code_lookup", technical_messages)
        self.assertNotIn("billing_field_check", technical_messages)
        self.assertIn("billing_field_check", billing_messages)
        self.assertNotIn("error_code_lookup", billing_messages)


if __name__ == "__main__":
    unittest.main()
