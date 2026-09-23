"""Upload and delete logic. No network: Qdrant and ingestion are stubbed.

Uploads are not persisted: the PDF is ingested from a temp file and discarded,
so these tests assert on what reaches Qdrant, not on files left behind.
"""

import pytest

import app.ingestion.upload as upload
from app.ingestion.upload import (
    DocumentNotFound,
    ProtectedDocument,
    UploadError,
    ingest_upload,
    safe_filename,
    validate,
)
from tests.pdf_fixture import write_pdf

PDF = b"%PDF-1.4 minimal"


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """Empty corpus, ingestion and Qdrant stubbed. Records what was ingested."""
    corpus = tmp_path / "raw_pdfs"
    corpus.mkdir()
    monkeypatch.setattr(upload, "RAW_PDF_DIR", corpus)

    seen = {"ingested": [], "cleared": []}

    def fake_ingest(path, chunker=None):
        # Capture the path so tests can assert on the source label and that the
        # temp file really exists at ingestion time.
        seen["ingested"].append((path.name, path.exists()))
        seen["last_dir"] = path.parent
        seen["chunker"] = chunker
        return {"file": path.name, "pages": 1, "chunks": 3}

    monkeypatch.setattr(upload, "ingest_path", fake_ingest)
    monkeypatch.setattr(upload, "clear", lambda name: seen["cleared"].append(name))
    monkeypatch.setattr(upload, "list_documents", lambda: {"documents": []})
    seen["corpus"] = corpus
    return seen


def _indexed(monkeypatch, *names):
    monkeypatch.setattr(
        upload,
        "list_documents",
        lambda: {"documents": [{"source": n, "chunks": 3} for n in names]},
    )


# --- name safety ---------------------------------------------------------

@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("../../.env.pdf", "env.pdf"),
        ("../../../etc/passwd.pdf", "passwd.pdf"),
        ("/absolute/path/doc.pdf", "doc.pdf"),
        ("weird ;rm -rf.pdf", "weird _rm -rf.pdf"),
        ("normal.pdf", "normal.pdf"),
    ],
)
def test_traversal_is_stripped(given, expected):
    assert safe_filename(given) == expected


@pytest.mark.parametrize("bad", ["notes.txt", "", "   ", "....pdf"])
def test_non_pdf_names_rejected(bad):
    with pytest.raises(UploadError):
        safe_filename(bad)


# --- content validation --------------------------------------------------

def test_rejects_non_pdf_bytes():
    with pytest.raises(UploadError, match="not a PDF"):
        validate(b"just text", "a.pdf")


def test_rejects_empty_and_oversize():
    with pytest.raises(UploadError, match="empty"):
        validate(b"", "a.pdf")
    with pytest.raises(UploadError, match="limit"):
        validate(b"%PDF-" + b"x" * upload.MAX_BYTES, "a.pdf")


# --- nothing is persisted ------------------------------------------------

def test_nothing_is_written_to_the_corpus(stubbed):
    ingest_upload(PDF, "new.pdf")
    assert list(stubbed["corpus"].iterdir()) == []


def test_temp_file_exists_during_ingestion_and_is_gone_after(stubbed):
    ingest_upload(PDF, "new.pdf")
    _name, existed_during = stubbed["ingested"][0]
    assert existed_during                       # readable while ingesting
    assert not stubbed["last_dir"].exists()     # temp dir cleaned up after


def test_source_label_is_the_real_filename_not_a_temp_name(stubbed):
    # load_pdf takes the citation label from path.name, so a mkstemp-style name
    # would surface next to every answer.
    ingest_upload(PDF, "../../NIS2 report.pdf")
    assert stubbed["ingested"][0][0] == "NIS2 report.pdf"


def test_uploads_use_the_fixed_chunker_whatever_chunker_is(stubbed, monkeypatch):
    # CHUNKER=structured is for the curated corpus; an upload's layout is unknown.
    monkeypatch.setenv("CHUNKER", "structured")
    ingest_upload(PDF, "new.pdf")
    assert stubbed["chunker"] == "fixed"


def test_temp_file_removed_even_when_ingestion_fails(stubbed, monkeypatch):
    dirs = []

    def boom(path, chunker=None):
        dirs.append(path.parent)
        raise RuntimeError("mistral 429")

    monkeypatch.setattr(upload, "ingest_path", boom)
    with pytest.raises(RuntimeError):
        ingest_upload(PDF, "doomed.pdf")
    assert not dirs[0].exists()


def test_scanned_pdf_yielding_no_text_is_rejected(stubbed, monkeypatch):
    monkeypatch.setattr(
        upload, "ingest_path", lambda p, chunker=None: {"file": p.name, "pages": 3, "chunks": 0}
    )
    with pytest.raises(UploadError, match="OCR"):
        ingest_upload(PDF, "scan.pdf")


# --- collisions ----------------------------------------------------------

def test_reupload_replaces_previous_chunks(stubbed, monkeypatch):
    _indexed(monkeypatch, "notes.pdf")
    ingest_upload(PDF, "notes.pdf")
    # Cleared first, so an edited PDF cannot leave orphans under the same label.
    assert stubbed["cleared"] == ["notes.pdf"]


def test_upload_cannot_shadow_a_curated_file(stubbed):
    write_pdf(stubbed["corpus"] / "CELEX.pdf", [["Article 1", "text"]])
    with pytest.raises(UploadError, match="curated corpus"):
        ingest_upload(PDF, "CELEX.pdf")


# --- delete --------------------------------------------------------------

def test_delete_clears_chunks(stubbed, monkeypatch):
    _indexed(monkeypatch, "temp.pdf")
    assert upload.delete_document("temp.pdf") == {"source": "temp.pdf", "chunks_removed": 3}
    assert stubbed["cleared"] == ["temp.pdf"]


def test_delete_refuses_curated_corpus_files(stubbed, monkeypatch):
    protected = write_pdf(stubbed["corpus"] / "CELEX.pdf", [["Article 1", "text"]])
    _indexed(monkeypatch, "CELEX.pdf")

    with pytest.raises(ProtectedDocument, match="curated corpus"):
        upload.delete_document("CELEX.pdf")

    assert protected.exists()          # file untouched
    assert stubbed["cleared"] == []    # and its vectors left alone


def test_delete_unknown_document_is_not_found(stubbed, monkeypatch):
    _indexed(monkeypatch)
    with pytest.raises(DocumentNotFound):
        upload.delete_document("ghost.pdf")


def test_delete_sanitises_the_name(stubbed, monkeypatch):
    _indexed(monkeypatch, "env.pdf")
    upload.delete_document("../../env.pdf")
    assert stubbed["cleared"] == ["env.pdf"]   # traversal stripped before use
