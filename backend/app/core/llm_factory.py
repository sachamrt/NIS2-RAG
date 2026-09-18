"""Chat-model factory.

Everything downstream (rag chain, agents, API) depends on the LangChain
``BaseChatModel`` interface only -- never on a vendor SDK. Swapping providers
is an .env change (``LLM_PROVIDER``), not a code change.

Provider SDKs are imported lazily inside each builder, so only the provider you
actually use has to be installed.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def _require_key(settings: Settings, field: str, provider: str) -> str:
    secret = getattr(settings, field)
    if secret is None or not secret.get_secret_value():
        raise ValueError(
            f"LLM_PROVIDER={provider} requires {field.upper()} to be set in .env"
        )
    return secret.get_secret_value()


def _build_mistral(settings: Settings) -> "BaseChatModel":
    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(
        model=settings.mistral_llm_model,
        api_key=_require_key(settings, "mistral_api_key", "mistral"),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


def _build_openai(settings: Settings) -> "BaseChatModel":
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_llm_model,
        api_key=_require_key(settings, "openai_api_key", "openai"),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


def _build_anthropic(settings: Settings) -> "BaseChatModel":
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.anthropic_llm_model,
        api_key=_require_key(settings, "anthropic_api_key", "anthropic"),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens or 4096,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


def _build_local(settings: Settings) -> "BaseChatModel":
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
    )


_BUILDERS: dict[str, Callable[[Settings], "BaseChatModel"]] = {
    "local": _build_local,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "mistral": _build_mistral,
}


def build_llm(provider: str | None = None, settings: Settings | None = None) -> "BaseChatModel":
    """Build a chat model. Defaults to the provider configured in .env."""
    settings = settings or get_settings()
    name = provider or settings.llm_provider
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown LLM provider {name!r}. Known: {', '.join(sorted(_BUILDERS))}"
        ) from None
    try:
        return builder(settings)
    except ImportError as exc:
        raise ImportError(
            f"Provider {name!r} is selected but its package is not installed: {exc}"
        ) from exc


@lru_cache
def get_llm() -> "BaseChatModel":
    """Cached chat model for the configured provider (the normal entrypoint)."""
    return build_llm()
