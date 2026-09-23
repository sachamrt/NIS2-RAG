"""Chunker factory -- same contract as the provider factories.

CHUNKER=fixed       ~1000-char recursive split within each page (phase 1)
CHUNKER=structured  one chunk per recital / Article paragraph / definition /
                    Annex row (app/ingestion/structure.py)

CHUNKER applies to the curated corpus (data/raw_pdfs/). Uploads are always
"fixed": their layout is unknown. Every chunk records the chunker that made it
(``metadata.chunker``), and corpus ingestion refuses to re-chunk the corpus in
place with a different chunker -- old and new chunks would both stay indexed.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.ingestion.splitter import CHUNK_OVERLAP, CHUNK_SIZE, split_documents
from app.ingestion.structure import split_structured

if TYPE_CHECKING:
    from langchain_core.documents import Document

Split = Callable[[list["Document"], int, int], tuple[list["Document"], list[str]]]


def _fixed(pages, chunk_size, chunk_overlap):
    return split_documents(pages, chunk_size, chunk_overlap)


def _structured(pages, chunk_size, chunk_overlap):
    # Sizes come from the document's own units; chunk_size does not apply.
    return split_structured(pages)


_BUILDERS: dict[str, Split] = {
    "fixed": _fixed,
    "structured": _structured,
}


def resolve_chunker(name: str | None = None) -> str:
    """The chunker to use: `name`, else CHUNKER from .env. Raises if unknown."""
    name = name or get_settings().chunker
    if name not in _BUILDERS:
        raise ValueError(f"Unknown chunker {name!r}. Known: {', '.join(sorted(_BUILDERS))}")
    return name


def split(
    pages: list["Document"],
    chunker: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[list["Document"], list[str]]:
    """Split one PDF's pages into (chunks, ids), tagging each with its chunker."""
    name = resolve_chunker(chunker)
    chunks, ids = _BUILDERS[name](pages, chunk_size, chunk_overlap)
    for chunk in chunks:
        chunk.metadata["chunker"] = name
    return chunks, ids
