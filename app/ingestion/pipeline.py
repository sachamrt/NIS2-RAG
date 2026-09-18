"""load -> split -> embed -> upsert.

Embedding is the slow, rate-limited, paid step, so chunks go up in batches and
each file is reported as it lands rather than at the very end.
"""

from dataclasses import dataclass, field
from pathlib import Path

from app.core.vectorstore import ensure_collection, get_vectorstore
from app.ingestion.loader import RAW_PDF_DIR, discover_pdfs, load_pdf
from app.ingestion.splitter import CHUNK_OVERLAP, CHUNK_SIZE, split_documents

BATCH_SIZE = 32


@dataclass
class IngestionReport:
    files: int = 0
    pages: int = 0
    chunks: int = 0
    per_file: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return f"{self.files} file(s), {self.pages} page(s), {self.chunks} chunk(s) upserted"


def ingest_path(
    path: Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size: int = BATCH_SIZE,
    on_progress=None,
) -> dict:
    """Ingest a single PDF. Returns a per-file record."""
    pages = load_pdf(path)
    chunks, ids = split_documents(pages, chunk_size, chunk_overlap)

    if chunks:
        store = get_vectorstore()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            store.add_documents(batch, ids=ids[start : start + batch_size])
            if on_progress:
                on_progress(path, min(start + batch_size, len(chunks)), len(chunks))

    return {"file": path.name, "pages": len(pages), "chunks": len(chunks)}


def ingest_directory(
    directory: Path | str = RAW_PDF_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size: int = BATCH_SIZE,
    on_progress=None,
) -> IngestionReport:
    """Ingest every PDF in `directory`. Safe to re-run: IDs are deterministic."""
    ensure_collection()  # fail fast on a dimension mismatch, before embedding
    report = IngestionReport()
    for path in discover_pdfs(directory):
        record = ingest_path(path, chunk_size, chunk_overlap, batch_size, on_progress)
        report.files += 1
        report.pages += record["pages"]
        report.chunks += record["chunks"]
        report.per_file.append(record)
    return report
