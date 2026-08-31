"""Provider-neutral async text generation for Anthropic and OpenAI APIs."""
from dataclasses import dataclass
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


def _text_response(text: str) -> Any:
    """Return the small Anthropic-compatible response shape used by the app."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)] if text else []
    )


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


class OpenAIMessageAdapter:
    """Expose OpenAI Responses/Chat Completions through ``messages.create``."""

    def __init__(self, client: Any, api_style: str):
        if api_style not in _SUPPORTED_OPENAI_STYLES:
            raise ValueError(f"unsupported OpenAI API style: {api_style}")
        self._client = client
        self._api_style = api_style
        self.messages = self

    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        **_: Any,
    ) -> Any:
        if self._api_style == "responses":
            kwargs: Dict[str, Any] = {
                "model": model,
                "input": messages,
                "max_output_tokens": max_tokens,
            }
            if system:
                kwargs["instructions"] = system
            # Reasoning models may reject temperature, so Responses calls omit it.
            response = await self._client.responses.create(**kwargs)
            return _text_response(_extract_responses_text(response))

        chat_messages = list(messages)
        if system:
            chat_messages = [{"role": "system", "content": system}] + chat_messages
        kwargs = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await self._client.chat.completions.create(**kwargs)
        return _text_response(_extract_chat_text(response))


def create_llm_client(config: LLMConfig) -> Any:
    """Create the selected SDK client, importing only the configured provider."""
    if config.provider == "anthropic":
        from anthropic import AsyncAnthropic

        kwargs: Dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return AsyncAnthropic(**kwargs)

    from openai import AsyncOpenAI

    kwargs = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAIMessageAdapter(AsyncOpenAI(**kwargs), config.api_style)
