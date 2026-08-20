import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.llm_client import OpenAIMessageAdapter, load_llm_config


class _Recorder:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class LLMConfigTests(unittest.TestCase):
    def test_anthropic_remains_default(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            config = load_llm_config()
        self.assertEqual(config.provider, "anthropic")
        self.assertEqual(config.api_style, "messages")

    def test_openai_config(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "test-model",
                "OPENAI_API_STYLE": "chat_completions",
                "OPENAI_BASE_URL": "https://example.test/v1",
            },
            clear=True,
        ):
            config = load_llm_config()
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.api_style, "chat_completions")


class OpenAIMessageAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_responses_mapping(self):
        recorder = _Recorder(SimpleNamespace(output_text="hello"))
        client = SimpleNamespace(responses=recorder)
        adapter = OpenAIMessageAdapter(client, "responses")

        response = await adapter.messages.create(
            model="test-model",
            system="be concise",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.1,
        )

        self.assertEqual(response.content[0].text, "hello")
        self.assertEqual(recorder.kwargs["instructions"], "be concise")
        self.assertEqual(recorder.kwargs["max_output_tokens"], 100)
        self.assertNotIn("temperature", recorder.kwargs)

    async def test_chat_completions_mapping(self):
        recorder = _Recorder(
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello from chat"))]
            )
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
        adapter = OpenAIMessageAdapter(client, "chat_completions")

        response = await adapter.messages.create(
            model="test-model",
            system="be concise",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.2,
        )

        self.assertEqual(response.content[0].text, "hello from chat")
        self.assertEqual(recorder.kwargs["messages"][0]["role"], "system")
        self.assertEqual(recorder.kwargs["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
