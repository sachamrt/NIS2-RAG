"""PDF discovery and loading.

One Document per page, carrying the metadata retrieval will later cite:
``source`` (file name) and ``page`` (1-based).
"""

from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document

RAW_PDF_DIR = Path("data/raw_pdfs")


def discover_pdfs(directory: Path | str = RAW_PDF_DIR) -> list[Path]:
    """Every .pdf under `directory`, recursively, in a stable order."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"PDF directory not found: {directory}")
    return sorted(p for p in directory.rglob("*.pdf") if p.is_file())


def load_pdf(path: Path) -> list[Document]:
    """Load one PDF into per-page Documents with normalised metadata.

    Uses pypdf directly rather than langchain-community's PyPDFLoader: the
    community package is being sunset, and all we need is page text. Pages that
    extract to nothing (scanned images) are skipped -- they would embed to
    noise. OCR, if ever needed, slots in here.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    total = len(reader.pages)
    docs: list[Document] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": number, "total_pages": total},
            )
        )
    return docs


def load_all(directory: Path | str = RAW_PDF_DIR) -> Iterator[tuple[Path, list[Document]]]:
    """Yield (path, pages) per PDF so callers can report progress per file."""
    for path in discover_pdfs(directory):
        yield path, load_pdf(path)
