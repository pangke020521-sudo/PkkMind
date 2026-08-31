import os
import unittest
from unittest.mock import patch

from agents.agent_orchestrator import AgentOrchestrator, AgentType
from core.llm_client import (
    LLMConfig,
    LLMRuntime,
    build_component_llm_runtimes,
    load_component_llm_config,
)


class ComponentLLMConfigTests(unittest.TestCase):
    def setUp(self):
        self.default = LLMConfig(
            provider="openai",
            api_key="global-openai-key",
            model="global-model",
            base_url="https://global.example/v1",
            api_style="responses",
        )

    def test_component_without_override_reuses_default_config(self):
        with patch.dict(os.environ, {}, clear=True):
            config = load_component_llm_config("technical", self.default)
        self.assertIs(config, self.default)

    def test_model_only_override_inherits_global_provider_settings(self):
        with patch.dict(
            os.environ,
            {"PKKMIND_TECHNICAL_LLM_MODEL": "technical-model"},
            clear=True,
        ):
            config = load_component_llm_config("technical", self.default)

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.api_key, "global-openai-key")
        self.assertEqual(config.model, "technical-model")
        self.assertEqual(config.base_url, "https://global.example/v1")
        self.assertEqual(config.api_style, "responses")

    def test_component_can_switch_from_openai_to_anthropic(self):
        with patch.dict(
            os.environ,
            {
                "PKKMIND_TECHNICAL_LLM_PROVIDER": "anthropic",
                "PKKMIND_TECHNICAL_LLM_MODEL": "claude-test",
                "PKKMIND_TECHNICAL_LLM_API_KEY": "component-key",
                "PKKMIND_TECHNICAL_LLM_BASE_URL": "https://anthropic.example",
            },
            clear=True,
        ):
            config = load_component_llm_config("technical", self.default)

        self.assertEqual(config.provider, "anthropic")
        self.assertEqual(config.api_key, "component-key")
        self.assertEqual(config.model, "claude-test")
        self.assertEqual(config.base_url, "https://anthropic.example")
        self.assertEqual(config.api_style, "messages")

    def test_runtime_builder_reuses_default_client_without_overrides(self):
        default_client = object()
        with patch.dict(os.environ, {}, clear=True):
            runtimes = build_component_llm_runtimes(
                ("general", "technical", "billing"),
                self.default,
                default_client,
            )

        self.assertTrue(all(runtime.client is default_client for runtime in runtimes.values()))

    def test_invalid_component_provider_is_rejected(self):
        with patch.dict(
            os.environ,
            {"PKKMIND_GENERAL_LLM_PROVIDER": "unsupported"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "anthropic 或 openai"):
                load_component_llm_config("general", self.default)


class AgentRuntimeAssignmentTests(unittest.TestCase):
    def test_orchestrator_assigns_override_to_only_target_agent(self):
        default_client = object()
        technical_client = object()
        technical_config = LLMConfig(
            provider="anthropic",
            api_key="test-key",
            model="technical-model",
        )

        orchestrator = AgentOrchestrator(
            client=default_client,
            model="default-model",
            embedding_enabled=False,
            provider="openai",
            agent_llms={
                AgentType.TECHNICAL: LLMRuntime(
                    config=technical_config,
                    client=technical_client,
                )
            },
        )

        general = orchestrator._pool[AgentType.GENERAL][0]
        technical = orchestrator._pool[AgentType.TECHNICAL][0]
        billing = orchestrator._pool[AgentType.BILLING][0]

        self.assertIs(general._client, default_client)
        self.assertEqual(general._provider, "openai")
        self.assertEqual(general._model, "default-model")
        self.assertIs(technical._client, technical_client)
        self.assertEqual(technical._provider, "anthropic")
        self.assertEqual(technical._model, "technical-model")
        self.assertIs(billing._client, default_client)

        stats = orchestrator.get_stats()
        self.assertEqual(stats["technical_0"]["provider"], "anthropic")
        self.assertEqual(stats["technical_0"]["model"], "technical-model")


if __name__ == "__main__":
    unittest.main()
