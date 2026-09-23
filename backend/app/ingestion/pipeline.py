"""load -> split -> embed -> upsert.

Embedding is the slow, rate-limited, paid step, so chunks go up in batches and
each file is reported as it lands rather than at the very end.
"""

from dataclasses import dataclass, field
from pathlib import Path

from app.core.vectorstore import ensure_collection, get_vectorstore, indexed_chunker
from app.ingestion.chunkers import resolve_chunker, split
from app.ingestion.loader import RAW_PDF_DIR, discover_pdfs, read_pdf
from app.ingestion.splitter import CHUNK_OVERLAP, CHUNK_SIZE

BATCH_SIZE = 32


@dataclass
class IngestionReport:
    files: int = 0
    pages: int = 0
    chunks: int = 0
    per_file: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return f"{self.files} file(s), {self.pages} page(s), {self.chunks} chunk(s) upserted"


def check_chunker(chunker: str, sources: list[str]) -> None:
    """Refuse to re-chunk the curated corpus with a different chunker in place.

    Chunk IDs differ between chunkers, so the old chunks would stay next to the
    new ones -- the same text indexed twice, and an eval of the collection that
    measures neither. Only the corpus is checked: uploads are always "fixed"
    and share the collection by design.
    """
    existing = indexed_chunker(sources)
    if existing and existing != chunker:
        raise ValueError(
            f"The corpus in this collection was chunked with {existing!r} but "
            f"CHUNKER={chunker!r}. Use a different QDRANT_COLLECTION or set "
            f"CHUNKER={existing}."
        )


def ingest_path(
    path: Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size: int = BATCH_SIZE,
    on_progress=None,
    *,
    chunker: str | None = None,
) -> dict:
    """Ingest a single PDF. Returns a per-file record.

    `chunker` defaults to CHUNKER, which is the curated corpus's chunker;
    uploads pass "fixed" explicitly.
    """
    loaded = read_pdf(path)
    pages = loaded.pages
    chunks, ids = split(
        pages, chunker=chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    if chunks:
        store = get_vectorstore()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            store.add_documents(batch, ids=ids[start : start + batch_size])
            if on_progress:
                on_progress(path, min(start + batch_size, len(chunks)), len(chunks))

    return {
        "file": path.name,
        "pages": len(pages),
        "chunks": len(chunks),
        "truncated_pages": loaded.truncated_pages,
    }


def ingest_directory(
    directory: Path | str = RAW_PDF_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size: int = BATCH_SIZE,
    on_progress=None,
) -> IngestionReport:
    """Ingest every PDF in `directory` with the corpus chunker (CHUNKER).

    Safe to re-run: IDs are deterministic.
    """
    ensure_collection()  # fail fast on a dimension mismatch, before embedding
    chunker = resolve_chunker()
    paths = discover_pdfs(directory)
    check_chunker(chunker, [path.name for path in paths])
    report = IngestionReport()
    for path in paths:
        record = ingest_path(
            path, chunk_size, chunk_overlap, batch_size, on_progress, chunker=chunker
        )
        report.files += 1
        report.pages += record["pages"]
        report.chunks += record["chunks"]
        report.per_file.append(record)
    return report
