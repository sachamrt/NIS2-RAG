"""Qdrant client and collection lifecycle.

The collection is created lazily with the vector size *probed from the
configured embedder*, so switching EMBEDDING_PROVIDER cannot silently produce a
dimension mismatch -- it either reuses a matching collection or refuses.

With a sparse model configured (SPARSE_PROVIDER), the collection also gets a
sparse vector slot and retrieval runs in hybrid mode. The sparse model runs
client-side; Qdrant only stores the vectors and applies the IDF weighting.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient, models

from app.core.config import Settings, get_settings
from app.core.embeddings_factory import get_embedding_dimension, get_embeddings
from app.core.sparse_factory import get_sparse_embeddings

if TYPE_CHECKING:
    from langchain_qdrant import QdrantVectorStore

# Name of the sparse vector slot, shared by collection creation and the
# LangChain store so the two cannot disagree (LangChain's own default).
SPARSE_VECTOR_NAME = "langchain-sparse"


@lru_cache
def _client_for_url(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def get_client(settings: Settings | None = None) -> QdrantClient:
    """One client per Qdrant URL (Settings itself is unhashable, so key on url)."""
    settings = settings or get_settings()
    return _client_for_url(settings.qdrant_url)


def ensure_collection(settings: Settings | None = None) -> str:
    """Create the collection if absent; verify its schema if present.

    Returns the collection name. Raises if an existing collection was built
    with a different embedder (wrong vector size) or lacks the sparse vector
    that hybrid search needs, which would otherwise fail confusingly at upsert
    or silently retrieve garbage.
    """
    settings = settings or get_settings()
    client = get_client(settings)
    name = settings.qdrant_collection
    dim = get_embedding_dimension()
    hybrid = get_sparse_embeddings() is not None

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            sparse_vectors_config=(
                {SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)}
                if hybrid
                else None
            ),
        )
        # Payload index: lets phase-2 filtering by source file stay fast.
        client.create_payload_index(
            collection_name=name,
            field_name="metadata.source",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        return name

    params = client.get_collection(name).config.params
    existing = params.vectors
    size = existing.size if hasattr(existing, "size") else existing[""].size
    if size != dim:
        raise ValueError(
            f"Collection {name!r} holds {size}-dim vectors but the configured "
            f"embedder produces {dim}-dim. Use a different QDRANT_COLLECTION "
            f"or drop the existing one."
        )
    if hybrid and SPARSE_VECTOR_NAME not in (params.sparse_vectors or {}):
        raise ValueError(
            f"Collection {name!r} has no sparse vector, but SPARSE_PROVIDER="
            f"{settings.sparse_provider!r} needs one for hybrid search. Use a "
            f"different QDRANT_COLLECTION or set SPARSE_PROVIDER=none."
        )
    return name


def get_vectorstore(settings: Settings | None = None) -> "QdrantVectorStore":
    """LangChain vectorstore bound to the configured collection and embedders.

    Hybrid (dense + sparse, RRF-fused) when a sparse model is configured,
    dense-only otherwise.
    """
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    settings = settings or get_settings()
    name = ensure_collection(settings)
    sparse = get_sparse_embeddings()
    return QdrantVectorStore(
        client=get_client(settings),
        collection_name=name,
        embedding=get_embeddings(),
        sparse_embedding=sparse,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        retrieval_mode=RetrievalMode.HYBRID if sparse else RetrievalMode.DENSE,
    )

    
def clear(source: str | None = None, settings: Settings | None = None) -> str:
    """Delete one source's chunks, or every point if `source` is None.

    Needed because deterministic IDs make re-ingestion idempotent but cannot
    remove chunks that no longer exist in an edited PDF.
    """
    settings = settings or get_settings()
    client = get_client(settings)
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        return "nothing (collection does not exist)"

    if source is None:
        client.delete_collection(name)
        ensure_collection(settings)
        return "all sources"

    client.delete(
        collection_name=name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source", match=models.MatchValue(value=source)
                    )
                ]
            )
        ),
    )
    return source


def collection_stats(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    client = get_client(settings)
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        return {"collection": name, "exists": False, "points": 0}
    info = client.get_collection(name)
    return {"collection": name, "exists": True, "points": info.points_count}
