"""PDF discovery and loading.

One Document per page, carrying the metadata retrieval will later cite:
``source`` (file name) and ``page`` (1-based).
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from app.core.paths import RAW_PDF_DIR

logger = logging.getLogger(__name__)

# pypdf stops extracting a page after this many form XObject invocations (a
# guard against malicious PDFs) and only says so in a log line. The rest of the
# page is silently missing from the index, so the loader listens for it.
_TRUNCATION_MARKER = "form XObject invocations"


@dataclass
class LoadedPdf:
    pages: list[Document]
    # 1-based numbers of pages whose text pypdf cut short.
    truncated_pages: list[int] = field(default_factory=list)


class _TruncationListener(logging.Handler):
    """Collects pypdf's truncation warnings raised by *this* thread.

    Uploads are ingested in FastAPI's threadpool, so two extractions can run at
    once; the thread filter keeps one upload from reporting another's pages.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.thread = threading.get_ident()
        self.fired = False

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread and _TRUNCATION_MARKER in record.getMessage():
            self.fired = True


@contextmanager
def _listen_for_truncation() -> Iterator[_TruncationListener]:
    pypdf_logger = logging.getLogger("pypdf")
    listener = _TruncationListener()
    pypdf_logger.addHandler(listener)
    try:
        yield listener
    finally:
        pypdf_logger.removeHandler(listener)


def discover_pdfs(directory: Path | str = RAW_PDF_DIR) -> list[Path]:
    """Every .pdf under `directory`, recursively, in a stable order."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"PDF directory not found: {directory}")
    return sorted(p for p in directory.rglob("*.pdf") if p.is_file())


def read_pdf(path: Path) -> LoadedPdf:
    """Load one PDF into per-page Documents, reporting truncated pages.

    Uses pypdf directly rather than langchain-community's PyPDFLoader: the
    community package is being sunset, and all we need is page text. Pages that
    extract to nothing (scanned images) are skipped -- they would embed to
    noise. OCR, if ever needed, slots in here.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    total = len(reader.pages)
    loaded = LoadedPdf(pages=[])
    for number, page in enumerate(reader.pages, start=1):
        with _listen_for_truncation() as listener:
            text = (page.extract_text() or "").strip()
        if listener.fired:
            loaded.truncated_pages.append(number)
            logger.warning(
                "%s p.%d: text extraction was cut short (too many form XObjects); "
                "the rest of the page is not indexed",
                path.name,
                number,
            )
        if not text:
            continue
        loaded.pages.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": number, "total_pages": total},
            )
        )
    return loaded


def load_pdf(path: Path) -> list[Document]:
    """Per-page Documents with normalised metadata (see ``read_pdf``)."""
    return read_pdf(path).pages


def load_all(directory: Path | str = RAW_PDF_DIR) -> Iterator[tuple[Path, list[Document]]]:
    """Yield (path, pages) per PDF so callers can report progress per file."""
    for path in discover_pdfs(directory):
        yield path, load_pdf(path)
