import asyncio
import json
import unittest
from types import SimpleNamespace

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentProfile,
    AgentType,
    BillingAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    TechnicalAgent,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel


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


def _request(**overrides):
    values = {
        "message": "登录时报401，并且订单被重复扣款99元",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.TECHNICAL_LOGIN,
        "intent_group": "technical",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {
            "error_code": ["401"],
            "amount": ["99元"],
            "order_id": [],
            "date": [],
        },
    }
    values.update(overrides)
    return Request(**values)


class AgentProfileTests(unittest.TestCase):
    def test_agents_have_distinct_structured_profiles(self):
        self.assertIsInstance(GeneralAgent.profile, AgentProfile)
        self.assertNotEqual(GeneralAgent.profile.role, TechnicalAgent.profile.role)
        self.assertNotEqual(TechnicalAgent.profile.workflow, BillingAgent.profile.workflow)
        self.assertLess(TechnicalAgent.profile.temperature, GeneralAgent.profile.temperature)
        self.assertEqual(BillingAgent.profile.temperature, 0.0)

    def test_domain_agents_build_distinct_role_packets(self):
        req = _request()
        general = json.loads(GeneralAgent(_FakeClient(), "model")._build_role_packet(req))
        technical = json.loads(TechnicalAgent(_FakeClient(), "model")._build_role_packet(req))
        billing = json.loads(BillingAgent(_FakeClient(), "model")._build_role_packet(req))

        self.assertIn("triage_targets", general)
        self.assertEqual(technical["diagnostic_fields"]["error_codes"], ["401"])
        self.assertIn("订单号或交易号", billing["verification_fields"]["missing_fields"])

    def test_profile_is_injected_and_controls_generation(self):
        client = _FakeClient("诊断完成")
        agent = TechnicalAgent(client, "technical-model", provider="openai")

        response = asyncio.run(agent.handle(_request()))

        self.assertTrue(response.success)
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["model"], "technical-model")
        self.assertEqual(call["max_tokens"], TechnicalAgent.profile.max_tokens)
        self.assertEqual(call["temperature"], TechnicalAgent.profile.temperature)
        self.assertIn("技术故障诊断与排障", call["system"])
        self.assertIn("角色输入契约", str(call["messages"]))

    def test_health_stats_expose_profile(self):
        orchestrator = AgentOrchestrator(
            client=_FakeClient(),
            model="default-model",
            embedding_enabled=False,
            provider="openai",
        )

        stats = orchestrator.get_stats()

        self.assertEqual(stats["technical_0"]["role"], "技术故障诊断与排障")
        self.assertEqual(stats["billing_0"]["temperature"], 0.0)
        self.assertIn("可能原因", stats["technical_0"]["output_contract"])


class EscalationAgentTests(unittest.TestCase):
    def test_escalation_agent_is_deterministic_and_does_not_call_llm(self):
        client = _FakeClient("不应使用此回答")
        agent = EscalationAgent(
            client,
            "deterministic",
            provider="none",
        )

        response = asyncio.run(agent.handle(_request(
            intent=IntentCategory.HUMAN_HANDOFF,
            intent_group="escalation",
            urgency=UrgencyLevel.HIGH,
        )))

        self.assertTrue(response.success)
        self.assertTrue(response.escalate)
        self.assertEqual(response.agent_type, AgentType.ESCALATION)
        self.assertEqual(client.calls, [])
        self.assertIn("尚未接入正式的人工客服", response.content)
        self.assertIn("不代表已经创建真实工单", response.content)

    def test_orchestrator_routes_critical_request_to_escalation_agent(self):
        client = _FakeClient("不应使用此回答")
        orchestrator = AgentOrchestrator(
            client=client,
            model="default-model",
            embedding_enabled=False,
            provider="openai",
        )

        result = asyncio.run(orchestrator.run(_request(
            intent=IntentCategory.TECHNICAL_CRASH,
            intent_group="technical",
            urgency=UrgencyLevel.CRITICAL,
        )))

        self.assertEqual(result.agent_type, AgentType.ESCALATION)
        self.assertEqual(result.primary_agent, AgentType.ESCALATION)
        self.assertTrue(result.escalated)
        self.assertEqual(client.calls, [])
        self.assertIn("CRITICAL", result.response)
        self.assertIn("尚未接入正式的人工客服", result.response)


if __name__ == "__main__":
    unittest.main()
