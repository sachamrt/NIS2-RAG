"""Sparse-embedding factory -- same contract as ``embeddings_factory``.

Sparse vectors (BM25) sit *next to* the dense embeddings, not instead of them:
with a sparse model configured, the vectorstore runs hybrid search (dense +
sparse, fused with RRF). ``none`` disables it and retrieval stays dense-only.

The sparse vector is part of the Qdrant collection schema, so changing
``SPARSE_PROVIDER`` means re-ingesting into a fresh collection.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_qdrant import SparseEmbeddings


def _build_none(settings: Settings) -> "SparseEmbeddings | None":
    return None


def _build_bm25(settings: Settings) -> "SparseEmbeddings | None":
    from langchain_qdrant import FastEmbedSparse

    # Runs locally; the model is downloaded on first use.
    return FastEmbedSparse(model_name=settings.sparse_model)


_BUILDERS: dict[str, Callable[[Settings], "SparseEmbeddings | None"]] = {
    "none": _build_none,
    "bm25": _build_bm25,
}


def build_sparse_embeddings(
    provider: str | None = None, settings: Settings | None = None
) -> "SparseEmbeddings | None":
    """Build a sparse model, or None for dense-only. Defaults to .env."""
    settings = settings or get_settings()
    name = provider or settings.sparse_provider
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown sparse provider {name!r}. Known: {', '.join(sorted(_BUILDERS))}"
        ) from None
    try:
        return builder(settings)
    except ImportError as exc:
        raise ImportError(
            f"Sparse provider {name!r} is selected but its package is not installed: {exc}"
        ) from exc


@lru_cache
def get_sparse_embeddings() -> "SparseEmbeddings | None":
    """Cached sparse model for the configured provider (the normal entrypoint)."""
    return build_sparse_embeddings()
