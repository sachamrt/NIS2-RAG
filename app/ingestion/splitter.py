"""Page Documents -> embeddable chunks with deterministic IDs.

The ID is a uuid5 of (source, page, chunk index, content hash). That makes
re-ingestion idempotent: an unchanged PDF upserts onto the same point IDs
instead of creating duplicates, and an edited page replaces its own chunks.
"""

import hashlib
import uuid
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from langchain_core.documents import Document

# Legal text: headings, articles, then sentences. Splitting on these in order
# keeps an article's text together far more often than a blind character split.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

_NAMESPACE = uuid.UUID("6f6b1f1e-3f9a-5c2b-9d4e-1a2b3c4d5e6f")


def build_splitter(
    chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
    )


def chunk_id(source: str, page: int, index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return str(uuid.uuid5(_NAMESPACE, f"{source}:{page}:{index}:{digest}"))


def split_documents(
    pages: list["Document"],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[list["Document"], list[str]]:
    """Split pages into chunks; return (chunks, ids) aligned by position."""
    chunks = build_splitter(chunk_size, chunk_overlap).split_documents(pages)

    ids: list[str] = []
    per_page_counter: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = int(chunk.metadata.get("page", 0))
        index = per_page_counter.get((source, page), 0)
        per_page_counter[(source, page)] = index + 1
        chunk.metadata["chunk_index"] = index
        ids.append(chunk_id(source, page, index, chunk.page_content))
    return chunks, ids
