"""Integrity of the evaluation set (backend/eval/nis2_eval.jsonl).

The eval set is only worth anything if its ground truth is right. These tests
check it against the real PDF: every evidence quote must appear verbatim on the
page it claims. A quote on the wrong page, or paraphrased, fails -- so a
mislabelled page can never silently skew the retrieval score.

Needs the corpus in data/raw_pdfs/; skipped (not failed) when it is absent.
"""

import json
import re
from functools import cache

import pytest

from app.core.paths import BACKEND_ROOT, RAW_PDF_DIR
from app.ingestion.loader import load_pdf

EVAL_FILE = BACKEND_ROOT / "eval" / "nis2_eval.jsonl"
TYPES = {
    "lookup", "definition", "multi_part", "multi_page",
    "comparison", "recital", "partial", "unanswerable",
}
# Types whose correct answer is a refusal, so they carry no evidence.
NO_EVIDENCE = {"unanswerable"}


def _load() -> list[dict]:
    return [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]


ENTRIES = _load()


def _norm(text: str) -> str:
    # PDF extraction breaks lines mid-sentence; compare on collapsed whitespace.
    return re.sub(r"\s+", " ", text).strip()


@cache
def _pages(source: str) -> dict[int, str]:
    path = RAW_PDF_DIR / source
    if not path.exists():
        pytest.skip(f"corpus file {source} not present")
    # Same loader as ingestion, so page numbers match what the system cites.
    return {d.metadata["page"]: _norm(d.page_content) for d in load_pdf(path)}


# --- schema --------------------------------------------------------------

def test_ids_are_unique():
    ids = [e["id"] for e in ENTRIES]
    assert len(ids) == len(set(ids)), "duplicate ids"


def test_size_is_within_target():
    assert 30 <= len(ENTRIES) <= 50


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["id"])
def test_entry_schema(entry):
    assert entry["type"] in TYPES
    assert entry["question"].strip() and entry["reference_answer"].strip()
    assert all(isinstance(g, list) and g for g in entry["must_include"])

    if entry["type"] in NO_EVIDENCE:
        assert entry["evidence"] == [], "a refusal has no supporting page"
    else:
        assert entry["evidence"], "answerable entries must cite their evidence"
    alts = entry.get("also_answered_by", [])
    if alts:
        assert entry["evidence"], "an alternative needs a primary answer to stand in for"
        primary = {(ev["source"], ev["page"]) for ev in entry["evidence"]}
        assert not primary & {(ev["source"], ev["page"]) for ev in alts}, (
            "a page cannot be both primary evidence and an alternative"
        )
    if entry["type"] == "multi_page":
        pages = {(ev["source"], ev["page"]) for ev in entry["evidence"]}
        assert len(pages) >= 2, "multi_page must span at least two pages"


# --- ground truth against the PDF ---------------------------------------

EVIDENCE = [
    (e["id"], ev) for e in ENTRIES for ev in e["evidence"] + e.get("also_answered_by", [])
]


def _evidence_id(value):
    return value if isinstance(value, str) else ""


@pytest.mark.parametrize(("entry_id", "ev"), EVIDENCE, ids=_evidence_id)
def test_quote_is_on_the_stated_page(entry_id, ev):
    pages = _pages(ev["source"])
    assert ev["page"] in pages, f"{entry_id}: page {ev['page']} does not exist"

    quote = _norm(ev["quote"])
    if quote in pages[ev["page"]]:
        return
    # Point at the right page if the quote is merely mislabelled.
    elsewhere = [p for p, text in pages.items() if quote in text]
    hint = f"it is on page(s) {elsewhere}" if elsewhere else "it is not in the document at all"
    pytest.fail(f"{entry_id}: quote not on page {ev['page']} -- {hint}")


def test_unanswerable_hallucination_markers_are_really_absent():
    # A must_not_include term that the directive actually contains would
    # penalise a correct answer. Guard against that.
    text = " ".join(_pages(ENTRIES[0]["evidence"][0]["source"]).values()).lower()
    for e in ENTRIES:
        if e["type"] != "unanswerable":
            continue
        for term in e["must_not_include"]:
            assert term.lower() not in text, f"{e['id']}: {term!r} is in the directive"
