"""Loader + splitter behaviour. No network: these must run in CI without keys."""

import pytest
from langchain_core.documents import Document

from app.ingestion.loader import discover_pdfs, load_pdf
from app.ingestion.splitter import chunk_id, split_documents
from tests.pdf_fixture import write_pdf

PAGES = [
    ["Article 1 - Subject matter", "This Directive lays down measures."],
    ["Article 21 - Risk management", "Entities shall take appropriate measures."],
]


@pytest.fixture
def pdf(tmp_path):
    return write_pdf(tmp_path / "corpus" / "nis2.pdf", PAGES)


def test_discover_finds_pdfs_recursively(pdf):
    assert discover_pdfs(pdf.parent.parent) == [pdf]


def test_discover_raises_on_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_pdfs(tmp_path / "nope")


def test_load_pdf_yields_one_document_per_page_with_citable_metadata(pdf):
    docs = load_pdf(pdf)
    assert len(docs) == 2
    assert [d.metadata["page"] for d in docs] == [1, 2]  # 1-based, not pypdf's 0-based
    assert {d.metadata["source"] for d in docs} == {"nis2.pdf"}
    assert "Subject matter" in docs[0].page_content


def test_blank_pages_are_skipped(tmp_path):
    path = write_pdf(tmp_path / "mixed.pdf", [["Real text here"], [], ["More text"]])
    pages = [d.metadata["page"] for d in load_pdf(path)]
    assert pages == [1, 3]  # page 2 extracted nothing and was dropped


def test_split_assigns_aligned_ids_and_chunk_indices(pdf):
    chunks, ids = split_documents(load_pdf(pdf), chunk_size=40, chunk_overlap=0)
    assert len(chunks) == len(ids) == len(set(ids))
    for chunk in chunks:
        assert chunk.metadata["source"] == "nis2.pdf"
        assert "chunk_index" in chunk.metadata
    first_page = [c.metadata["chunk_index"] for c in chunks if c.metadata["page"] == 1]
    assert first_page == list(range(len(first_page)))  # per-page, contiguous, from 0


def test_ids_are_stable_across_runs(pdf):
    _, first = split_documents(load_pdf(pdf), chunk_size=40, chunk_overlap=0)
    _, second = split_documents(load_pdf(pdf), chunk_size=40, chunk_overlap=0)
    assert first == second  # re-ingestion upserts instead of duplicating


def test_id_changes_when_content_changes():
    unchanged = chunk_id("nis2.pdf", 1, 0, "original text")
    assert chunk_id("nis2.pdf", 1, 0, "original text") == unchanged
    assert chunk_id("nis2.pdf", 1, 0, "edited text") != unchanged
    assert chunk_id("other.pdf", 1, 0, "original text") != unchanged


def test_overlap_carries_context_between_chunks():
    long_text = " ".join(f"sentence{i}." for i in range(80))
    doc = Document(page_content=long_text, metadata={"source": "a.pdf", "page": 1})
    chunks, _ = split_documents([doc], chunk_size=120, chunk_overlap=40)
    assert len(chunks) > 1
    tail = chunks[0].page_content[-20:]
    assert any(word in chunks[1].page_content for word in tail.split())
