import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.llm_client import AnthropicMessageAdapter, OpenAIMessageAdapter, load_llm_config


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

    async def test_chat_completions_normalizes_tool_calls_and_results(self):
        recorder = _Recorder(SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        id="call_1",
                        function=SimpleNamespace(
                            name="error_code_lookup",
                            arguments='{"error_code":"401"}',
                        ),
                    )],
                ),
            )],
        ))
        client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
        adapter = OpenAIMessageAdapter(client, "chat_completions")

        response = await adapter.messages.create(
            model="test-model",
            messages=[{"role": "user", "content": "401是什么"}],
            max_tokens=100,
            tools=[{
                "name": "error_code_lookup",
                "description": "lookup",
                "parameters": {"type": "object", "properties": {}},
            }],
            tool_choice="error_code_lookup",
        )

        self.assertEqual(response.tool_calls[0].name, "error_code_lookup")
        self.assertEqual(response.tool_calls[0].arguments, {"error_code": "401"})
        self.assertEqual(
            recorder.kwargs["tool_choice"]["function"]["name"],
            "error_code_lookup",
        )

        recorder.response = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="done", tool_calls=[]),
            )],
        )
        await adapter.messages.create(
            model="test-model",
            messages=[
                {"role": "assistant", "content": "", "tool_calls": response.tool_calls},
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ],
            max_tokens=100,
        )
        self.assertEqual(recorder.kwargs["messages"][0]["role"], "assistant")
        self.assertEqual(recorder.kwargs["messages"][1]["role"], "tool")

    async def test_responses_normalizes_function_call_and_output(self):
        recorder = _Recorder(SimpleNamespace(
            output_text="",
            status="completed",
            output=[{
                "type": "function_call",
                "call_id": "call_r1",
                "name": "billing_field_check",
                "arguments": '{"entities":{}}',
            }],
        ))
        adapter = OpenAIMessageAdapter(SimpleNamespace(responses=recorder), "responses")

        response = await adapter.messages.create(
            model="test-model",
            messages=[{"role": "user", "content": "check"}],
            max_tokens=100,
            tools=[{
                "name": "billing_field_check",
                "description": "check",
                "parameters": {"type": "object", "properties": {}},
            }],
        )
        self.assertEqual(response.tool_calls[0].id, "call_r1")

        recorder.response = SimpleNamespace(output_text="done", status="completed", output=[])
        await adapter.messages.create(
            model="test-model",
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": response.tool_calls,
                    "provider_items": response.provider_items,
                },
                {"role": "tool", "tool_call_id": "call_r1", "content": "result"},
            ],
            max_tokens=100,
        )
        self.assertEqual(recorder.kwargs["input"][0]["type"], "function_call")
        self.assertEqual(recorder.kwargs["input"][1]["type"], "function_call_output")


class AnthropicMessageAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_normalizes_tool_use_and_tool_result(self):
        recorder = _Recorder(SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(
                type="tool_use",
                id="toolu_1",
                name="error_code_lookup",
                input={"error_code": "401"},
            )],
        ))
        adapter = AnthropicMessageAdapter(SimpleNamespace(messages=recorder))

        response = await adapter.messages.create(
            model="claude-test",
            messages=[{"role": "user", "content": "401是什么"}],
            max_tokens=100,
            tools=[{
                "name": "error_code_lookup",
                "description": "lookup",
                "parameters": {"type": "object", "properties": {}},
            }],
            tool_choice="error_code_lookup",
        )
        self.assertEqual(response.tool_calls[0].arguments, {"error_code": "401"})
        self.assertEqual(recorder.kwargs["tool_choice"], {
            "type": "tool",
            "name": "error_code_lookup",
        })

        recorder.response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="done")],
        )
        await adapter.messages.create(
            model="claude-test",
            messages=[
                {"role": "assistant", "content": "", "tool_calls": response.tool_calls},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "result"},
            ],
            max_tokens=100,
        )
        self.assertEqual(recorder.kwargs["messages"][0]["role"], "assistant")
        self.assertEqual(
            recorder.kwargs["messages"][1]["content"][0]["type"],
            "tool_result",
        )


if __name__ == "__main__":
    unittest.main()
