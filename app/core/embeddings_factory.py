"""Embedding-model factory -- same contract as ``llm_factory``.

Downstream code depends on the LangChain ``Embeddings`` interface only.
Note that the vector dimension is provider-specific (mistral-embed is 1024,
text-embedding-3-small is 1536, nomic-embed-text is 768), so a Qdrant
collection is only valid for the provider that populated it: changing
``EMBEDDING_PROVIDER`` means re-ingesting into a fresh collection.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import Settings, get_settings
from app.core.llm_factory import _require_key

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def _build_mistral(settings: Settings) -> "Embeddings":
    from langchain_mistralai import MistralAIEmbeddings

    return MistralAIEmbeddings(
        model=settings.mistral_embedding_model,
        api_key=_require_key(settings, "mistral_api_key", "mistral"),
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout,
    )


def _build_openai(settings: Settings) -> "Embeddings":
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=_require_key(settings, "openai_api_key", "openai"),
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout,
    )


def _build_local(settings: Settings) -> "Embeddings":
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


_BUILDERS: dict[str, Callable[[Settings], "Embeddings"]] = {
    "local": _build_local,
    "openai": _build_openai,
    "mistral": _build_mistral,
}


def build_embeddings(
    provider: str | None = None, settings: Settings | None = None
) -> "Embeddings":
    """Build an embedding model. Defaults to the provider configured in .env."""
    settings = settings or get_settings()
    name = provider or settings.embedding_provider
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown embedding provider {name!r}. Known: {', '.join(sorted(_BUILDERS))}"
        ) from None
    try:
        return builder(settings)
    except ImportError as exc:
        raise ImportError(
            f"Provider {name!r} is selected but its package is not installed: {exc}"
        ) from exc


@lru_cache
def get_embeddings() -> "Embeddings":
    """Cached embedding model for the configured provider (the normal entrypoint)."""
    return build_embeddings()


@lru_cache
def get_embedding_dimension() -> int:
    """Vector size of the configured embedder, probed at runtime.

    Used when creating the Qdrant collection so no dimension table has to be
    kept in sync with the providers.
    """
    return len(get_embeddings().embed_query("dimension probe"))
