"""Accept an uploaded PDF, store it, and ingest it.

Kept out of the route layer: the API only translates ``UploadError`` into a
status code. Everything a client sends -- file name, declared content type,
size -- is treated as hostile until checked here.
"""

import logging
import re
import tempfile
from pathlib import Path

from app.core.paths import RAW_PDF_DIR
from app.core.vectorstore import clear
from app.ingestion.pipeline import ingest_path
from app.rag.documents import list_documents

logger = logging.getLogger(__name__)

MAX_BYTES = 50 * 1024 * 1024  # 50 MB
PDF_MAGIC = b"%PDF-"
MAX_NAME_LENGTH = 120
MAX_COLLISION_SUFFIX = 1000

# Anything outside this set is replaced, so a stored name can never carry path
# separators, control characters, or shell-significant punctuation.
_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]")


class UploadError(ValueError):
    """Rejected upload. The message is safe to return to the client."""


def safe_filename(name: str) -> str:
    """Reduce a client-supplied name to a bare, safe ``*.pdf`` file name.

    ``Path(name).name`` drops any directory part, so "../../.env" becomes
    ".env" and "C:\\evil\\x.pdf" becomes "x.pdf" -- the traversal is gone before
    the name is ever joined to a directory.
    """
    bare = Path(name.strip()).name
    bare = _UNSAFE.sub("_", bare).strip(" .")

    if not bare:
        raise UploadError("File name is empty after sanitisation.")
    if not bare.lower().endswith(".pdf"):
        raise UploadError("Only .pdf files are accepted.")
    if len(bare) > MAX_NAME_LENGTH:
        stem, suffix = bare[: -len(".pdf")], ".pdf"
        bare = stem[: MAX_NAME_LENGTH - len(suffix)] + suffix
    return bare


def validate(content: bytes, name: str) -> str:
    """Check the bytes really are a PDF and return the safe file name.

    The extension and the declared Content-Type are both client-controlled, so
    the magic bytes are the only trustworthy signal of what this actually is.
    """
    filename = safe_filename(name)

    if not content:
        raise UploadError("File is empty.")
    if len(content) > MAX_BYTES:
        mb = len(content) / 1024 / 1024
        raise UploadError(f"File is {mb:.1f} MB; the limit is {MAX_BYTES // 1024 // 1024} MB.")
    if not content.startswith(PDF_MAGIC):
        raise UploadError("File is not a PDF (missing %PDF- header).")
    return filename


def ingest_upload(content: bytes, name: str) -> dict:
    """Validate, ingest, and keep nothing. Returns ``ingest_path``'s record:
    {"file", "pages", "chunks"}.

    The PDF is written to a temp directory only because the loader reads from a
    path, and is removed as soon as ingestion finishes. Once the chunks are in
    Qdrant the file has no further use: rebuilding the index from scratch is
    expected to lose uploaded documents, and that is the accepted trade.

    It is written under its real name rather than a random temp name because
    ``load_pdf`` takes the source label from ``path.name`` -- a mkstemp name
    would end up as the citation shown next to every answer.
    """
    filename = validate(content, name)

    if (RAW_PDF_DIR / filename).exists():
        raise UploadError(
            f"{filename!r} already exists in the curated corpus (data/raw_pdfs/). "
            "Rename the upload, or re-ingest the corpus instead."
        )

    # Re-uploading a name that is already indexed replaces it: without this an
    # edited PDF would leave its previous chunks orphaned under the same label,
    # since chunk IDs include a content hash.
    if filename in {d["source"] for d in list_documents()["documents"]}:
        logger.info("replacing previously indexed %s", filename)
        clear(filename)

    with tempfile.TemporaryDirectory(prefix="nis2-upload-") as tmp:
        path = Path(tmp) / filename
        path.write_bytes(content)
        record = ingest_path(path)

    if record["chunks"] == 0:
        raise UploadError(
            f"{filename} produced no text -- it is probably a scanned PDF, "
            "which needs OCR before it can be indexed."
        )

    logger.info("ingested %s: %s pages, %s chunks", filename, record["pages"], record["chunks"])
    return record


class DocumentNotFound(LookupError):
    """No such document, by name. The message is safe to return to the client."""


class ProtectedDocument(PermissionError):
    """The document is part of the hand-curated corpus and is not API-deletable."""


def delete_document(filename: str) -> dict:
    """Remove an uploaded document's chunks from Qdrant.

    Uploads leave no file behind, so deleting one is purely a vector clear.
    A document whose PDF sits in data/raw_pdfs/ is refused: the app did not put
    that file there and will not remove it, and clearing only its chunks would
    be undone by the next ``ingest_cli.py`` run anyway.
    """
    filename = safe_filename(filename)

    if (RAW_PDF_DIR / filename).exists():
        raise ProtectedDocument(
            f"{filename!r} is part of the curated corpus in data/raw_pdfs/ and "
            "cannot be deleted from the app. Remove the file yourself, then run "
            "scripts/ingest_cli.py --clear to drop its chunks."
        )

    indexed = {d["source"]: d["chunks"] for d in list_documents()["documents"]}
    if filename not in indexed:
        raise DocumentNotFound(f"No uploaded document named {filename!r}.")

    chunks = indexed[filename]
    clear(filename)
    logger.info("deleted %s: %s chunks removed", filename, chunks)
    return {"source": filename, "chunks_removed": chunks}
