"""Provider-neutral async text generation for Anthropic and OpenAI APIs."""
from dataclasses import dataclass, field
import json
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional


_SUPPORTED_PROVIDERS = {"anthropic", "openai"}
_SUPPORTED_OPENAI_STYLES = {"responses", "chat_completions"}


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    model: str
    base_url: Optional[str] = None
    api_style: str = "messages"


@dataclass(frozen=True)
class LLMRuntime:
    """A validated configuration paired with its provider SDK client."""

    config: LLMConfig
    client: Any


@dataclass(frozen=True)
class UnifiedToolCall:
    """Provider-neutral structured request to execute one local function tool."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class UnifiedLLMResponse:
    """Minimal response shape consumed by PkkMind across all providers."""

    content: List[Any]
    tool_calls: List[UnifiedToolCall]
    stop_reason: Optional[str] = None
    provider_items: List[Any] = field(default_factory=list)
    raw: Any = None


def load_llm_config() -> LLMConfig:
    """Load and validate the selected text-generation provider from the environment."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise RuntimeError(
            f"不支持的 LLM_PROVIDER: {provider!r}，可选值为 anthropic 或 openai"
        )

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic，但未设置 ANTHROPIC_API_KEY")
        return LLMConfig(
            provider=provider,
            api_key=api_key,
            model=os.getenv(
                "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
            ).strip(),
            base_url=os.getenv("ANTHROPIC_BASE_URL", "").strip() or None,
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    api_style = os.getenv("OPENAI_API_STYLE", "responses").strip().lower()
    if not api_key:
        raise RuntimeError("LLM_PROVIDER=openai，但未设置 OPENAI_API_KEY")
    if not model:
        raise RuntimeError("LLM_PROVIDER=openai，但未设置 OPENAI_MODEL")
    if api_style not in _SUPPORTED_OPENAI_STYLES:
        raise RuntimeError("OPENAI_API_STYLE 仅支持 responses 或 chat_completions")
    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
        api_style=api_style,
    )


def load_component_llm_config(component: str, default: LLMConfig) -> LLMConfig:
    """Load an optional component-specific LLM override.

    Components use provider-neutral environment variables such as
    ``PKKMIND_TECHNICAL_LLM_PROVIDER`` and ``PKKMIND_TECHNICAL_LLM_MODEL``.
    Missing values inherit the global configuration when the provider stays the
    same, or the selected provider's normal global variables when it changes.
    """

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", component).strip("_").upper()
    if not normalized:
        raise ValueError("component 不能为空")
    prefix = f"PKKMIND_{normalized}_LLM_"
    override_keys = (
        "PROVIDER",
        "API_KEY",
        "MODEL",
        "BASE_URL",
        "API_STYLE",
    )
    if not any(f"{prefix}{key}" in os.environ for key in override_keys):
        return default

    provider = os.getenv(f"{prefix}PROVIDER", default.provider).strip().lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise RuntimeError(
            f"不支持的 {prefix}PROVIDER: {provider!r}，可选值为 anthropic 或 openai"
        )

    same_provider = provider == default.provider
    if provider == "anthropic":
        api_key = os.getenv(f"{prefix}API_KEY", "").strip()
        if not api_key:
            api_key = default.api_key if same_provider else os.getenv("ANTHROPIC_API_KEY", "").strip()
        model = os.getenv(f"{prefix}MODEL", "").strip()
        if not model:
            model = default.model if same_provider else os.getenv(
                "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
            ).strip()
        base_url_key = f"{prefix}BASE_URL"
        if base_url_key in os.environ:
            base_url = os.getenv(base_url_key, "").strip() or None
        else:
            base_url = default.base_url if same_provider else os.getenv(
                "ANTHROPIC_BASE_URL", ""
            ).strip() or None
        if not api_key:
            raise RuntimeError(f"{prefix}PROVIDER=anthropic，但未设置可用的 API Key")
        return LLMConfig(
            provider="anthropic",
            api_key=api_key,
            model=model,
            base_url=base_url,
            api_style="messages",
        )

    api_key = os.getenv(f"{prefix}API_KEY", "").strip()
    if not api_key:
        api_key = default.api_key if same_provider else os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv(f"{prefix}MODEL", "").strip()
    if not model:
        model = default.model if same_provider else os.getenv("OPENAI_MODEL", "").strip()
    base_url_key = f"{prefix}BASE_URL"
    if base_url_key in os.environ:
        base_url = os.getenv(base_url_key, "").strip() or None
    else:
        base_url = default.base_url if same_provider else os.getenv(
            "OPENAI_BASE_URL", ""
        ).strip() or None
    api_style = os.getenv(f"{prefix}API_STYLE", "").strip().lower()
    if not api_style:
        api_style = default.api_style if same_provider else os.getenv(
            "OPENAI_API_STYLE", "responses"
        ).strip().lower()
    if not api_key:
        raise RuntimeError(f"{prefix}PROVIDER=openai，但未设置可用的 API Key")
    if not model:
        raise RuntimeError(f"{prefix}PROVIDER=openai，但未设置可用的模型名称")
    if api_style not in _SUPPORTED_OPENAI_STYLES:
        raise RuntimeError(
            f"{prefix}API_STYLE 仅支持 responses 或 chat_completions"
        )
    return LLMConfig(
        provider="openai",
        api_key=api_key,
        model=model,
        base_url=base_url,
        api_style=api_style,
    )


def build_component_llm_runtimes(
    components: Iterable[str],
    default_config: LLMConfig,
    default_client: Any,
) -> Dict[str, LLMRuntime]:
    """Create component runtimes while reusing clients for identical configs."""

    clients: Dict[LLMConfig, Any] = {default_config: default_client}
    runtimes: Dict[str, LLMRuntime] = {}
    for component in components:
        config = load_component_llm_config(component, default_config)
        client = clients.get(config)
        if client is None:
            client = create_llm_client(config)
            clients[config] = client
        runtimes[component] = LLMRuntime(config=config, client=client)
    return runtimes


def _unified_response(
    text: str,
    *,
    tool_calls: Optional[List[UnifiedToolCall]] = None,
    stop_reason: Optional[str] = None,
    provider_items: Optional[List[Any]] = None,
    raw: Any = None,
) -> UnifiedLLMResponse:
    """Return the Anthropic-like text blocks historically consumed by the app."""
    return UnifiedLLMResponse(
        content=[SimpleNamespace(type="text", text=text)] if text else [],
        tool_calls=list(tool_calls or []),
        stop_reason=stop_reason,
        provider_items=list(provider_items or []),
        raw=raw,
    )


def _parse_arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _tool_call_dict(call: Any) -> Dict[str, Any]:
    if isinstance(call, UnifiedToolCall):
        return {"id": call.id, "name": call.name, "arguments": call.arguments}
    if isinstance(call, dict):
        return {
            "id": str(call.get("id", "")),
            "name": str(call.get("name", "")),
            "arguments": _parse_arguments(call.get("arguments", {})),
        }
    return {
        "id": str(getattr(call, "id", "")),
        "name": str(getattr(call, "name", "")),
        "arguments": _parse_arguments(getattr(call, "arguments", {})),
    }


def _chat_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            tool_calls = []
            for item in message["tool_calls"]:
                call = _tool_call_dict(item)
                tool_calls.append({
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    },
                })
            converted.append({
                "role": "assistant",
                "content": message.get("content") or None,
                "tool_calls": tool_calls,
            })
        elif role == "tool":
            converted.append({
                "role": "tool",
                "tool_call_id": message.get("tool_call_id", ""),
                "content": str(message.get("content", "")),
            })
        else:
            converted.append(dict(message))
    return converted


def _responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            provider_items = message.get("provider_items") or []
            if provider_items:
                converted.extend(provider_items)
                continue
            if message.get("content"):
                converted.append({"role": "assistant", "content": message["content"]})
            for item in message["tool_calls"]:
                call = _tool_call_dict(item)
                converted.append({
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                })
        elif role == "tool":
            converted.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id", ""),
                "output": str(message.get("content", "")),
            })
        else:
            converted.append(dict(message))
    return converted


def _anthropic_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    pending_results: List[Dict[str, Any]] = []

    def flush_results() -> None:
        nonlocal pending_results
        if pending_results:
            converted.append({"role": "user", "content": pending_results})
            pending_results = []

    for message in messages:
        role = message.get("role")
        if role == "tool":
            pending_results.append({
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": str(message.get("content", "")),
            })
            continue
        flush_results()
        if role == "assistant" and message.get("tool_calls"):
            content: List[Dict[str, Any]] = []
            if message.get("content"):
                content.append({"type": "text", "text": str(message["content"])})
            for item in message["tool_calls"]:
                call = _tool_call_dict(item)
                content.append({
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["arguments"],
                })
            converted.append({"role": "assistant", "content": content})
        else:
            converted.append(dict(message))
    flush_results()
    return converted


def _extract_responses_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    texts: List[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            text = getattr(content, "text", None)
            if isinstance(content, dict):
                text = content.get("text", text)
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def _extract_chat_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if isinstance(content, str):
        return content

    texts: List[str] = []
    for part in content or []:
        text = getattr(part, "text", None)
        if isinstance(part, dict):
            text = part.get("text", text)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)


def _extract_chat_tool_calls(response: Any) -> List[UnifiedToolCall]:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    calls: List[UnifiedToolCall] = []
    for item in getattr(message, "tool_calls", None) or []:
        function = getattr(item, "function", None)
        if isinstance(item, dict):
            function = item.get("function", function)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if isinstance(function, dict):
            name = function.get("name", name)
            arguments = function.get("arguments", arguments)
        call_id = getattr(item, "id", None)
        if isinstance(item, dict):
            call_id = item.get("id", call_id)
        if name:
            calls.append(UnifiedToolCall(
                id=str(call_id or f"call_{len(calls)}"),
                name=str(name),
                arguments=_parse_arguments(arguments),
            ))
    return calls


def _extract_responses_tool_calls(response: Any) -> List[UnifiedToolCall]:
    calls: List[UnifiedToolCall] = []
    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None)
        if isinstance(item, dict):
            item_type = item.get("type", item_type)
        if item_type != "function_call":
            continue
        name = getattr(item, "name", None)
        arguments = getattr(item, "arguments", None)
        call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
        if isinstance(item, dict):
            name = item.get("name", name)
            arguments = item.get("arguments", arguments)
            call_id = item.get("call_id", item.get("id", call_id))
        if name:
            calls.append(UnifiedToolCall(
                id=str(call_id or f"call_{len(calls)}"),
                name=str(name),
                arguments=_parse_arguments(arguments),
            ))
    return calls


def _responses_provider_items(response: Any) -> List[Dict[str, Any]]:
    """Preserve Responses reasoning/function items required for stateless tool turns."""
    items: List[Dict[str, Any]] = []
    for item in getattr(response, "output", None) or []:
        if isinstance(item, dict):
            items.append(dict(item))
            continue
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                items.append(dumped)
    return items


class AnthropicMessageAdapter:
    """Normalize Anthropic Messages text and ``tool_use`` blocks."""

    supports_tool_calling = True
    supports_named_tool_choice = True

    def __init__(self, client: Any):
        self._client = client
        self.messages = self

    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **_: Any,
    ) -> UnifiedLLMResponse:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": _anthropic_messages(messages),
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {"type": "object"}),
                }
                for tool in tools
            ]
            if tool_choice in ("auto", "required"):
                kwargs["tool_choice"] = {"type": "auto" if tool_choice == "auto" else "any"}
            elif tool_choice and tool_choice != "none":
                kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        response = await self._client.messages.create(**kwargs)
        text_parts: List[str] = []
        tool_calls: List[UnifiedToolCall] = []
        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", None)
            if isinstance(block, dict):
                block_type = block.get("type", block_type)
            if block_type == "text":
                text = getattr(block, "text", None)
                if isinstance(block, dict):
                    text = block.get("text", text)
                if isinstance(text, str):
                    text_parts.append(text)
            elif block_type == "tool_use":
                call_id = getattr(block, "id", None)
                name = getattr(block, "name", None)
                arguments = getattr(block, "input", None)
                if isinstance(block, dict):
                    call_id = block.get("id", call_id)
                    name = block.get("name", name)
                    arguments = block.get("input", arguments)
                if name:
                    tool_calls.append(UnifiedToolCall(
                        id=str(call_id or f"call_{len(tool_calls)}"),
                        name=str(name),
                        arguments=_parse_arguments(arguments),
                    ))
        return _unified_response(
            "\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=getattr(response, "stop_reason", None),
            raw=response,
        )


class OpenAIMessageAdapter:
    """Expose OpenAI Responses/Chat Completions through ``messages.create``."""

    def __init__(self, client: Any, api_style: str):
        if api_style not in _SUPPORTED_OPENAI_STYLES:
            raise ValueError(f"unsupported OpenAI API style: {api_style}")
        self._client = client
        self._api_style = api_style
        self.messages = self
        self.supports_tool_calling = True
        self.supports_named_tool_choice = True

    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **_: Any,
    ) -> Any:
        if self._api_style == "responses":
            kwargs: Dict[str, Any] = {
                "model": model,
                "input": _responses_input(messages),
                "max_output_tokens": max_tokens,
            }
            if system:
                kwargs["instructions"] = system
            if tools:
                kwargs["parallel_tool_calls"] = False
                kwargs["tools"] = [
                    {
                        "type": "function",
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object"}),
                    }
                    for tool in tools
                ]
                if tool_choice in ("auto", "required", "none"):
                    kwargs["tool_choice"] = tool_choice
                elif tool_choice:
                    kwargs["tool_choice"] = {"type": "function", "name": tool_choice}
            # Reasoning models may reject temperature, so Responses calls omit it.
            response = await self._client.responses.create(**kwargs)
            return _unified_response(
                _extract_responses_text(response),
                tool_calls=_extract_responses_tool_calls(response),
                stop_reason=getattr(response, "status", None),
                provider_items=_responses_provider_items(response),
                raw=response,
            )

        chat_messages = _chat_messages(messages)
        if system:
            chat_messages = [{"role": "system", "content": system}] + chat_messages
        kwargs = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["parallel_tool_calls"] = False
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object"}),
                    },
                }
                for tool in tools
            ]
            if tool_choice in ("auto", "required", "none"):
                kwargs["tool_choice"] = tool_choice
            elif tool_choice:
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }
        response = await self._client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
        return _unified_response(
            _extract_chat_text(response),
            tool_calls=_extract_chat_tool_calls(response),
            stop_reason=finish_reason,
            raw=response,
        )


def create_llm_client(config: LLMConfig) -> Any:
    """Create the selected SDK client, importing only the configured provider."""
    if config.provider == "anthropic":
        from anthropic import AsyncAnthropic

        kwargs: Dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return AnthropicMessageAdapter(AsyncAnthropic(**kwargs))

    from openai import AsyncOpenAI

    kwargs = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAIMessageAdapter(AsyncOpenAI(**kwargs), config.api_style)
