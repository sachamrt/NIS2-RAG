"""Structure-aware chunking. No network; the real-PDF test skips if it is absent.

The synthetic pages reproduce line patterns seen in the NIS2 PDF, each one a
trap for a naive parser: footnotes numbered like recitals, a wrapped line that
starts with "(6)", "(i)" meaning a letter in one place and a roman in another.
"""

import pytest
from langchain_core.documents import Document

from app.core.paths import RAW_PDF_DIR
from app.evaluation.metrics import retrieval_metrics
from app.ingestion.chunkers import split
from app.ingestion.structure import document_lines, parse, pieces, split_structured

FOOTER = "EN Official Journal of the European Union L 333/80 27.12.2022"

PAGE_1 = """DIRECTIVE (EU) 2022/2555 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL
Whereas:
(1) Directive (EU) 2016/1148 of the European Parliament and the Council (4) aimed to build
cybersecurity capabilities across the Union.
(2) The legal basis was Article 114 of the Treaty on the Functioning of the European Union
(TFEU), the objective of which is the internal market.
(1) OJ C 233, 16.6.2022, p. 22.
(2) Directive (EU) 2016/1148 of the European Parliament (OJ L 194, 19.7.2016, p. 1).
""" + FOOTER

PAGE_2 = """HAVE ADOPTED THIS DIRECTIVE:
CHAPTER I
GENERAL PROVISIONS
Article 1
Subject matter
1. This Directive lays down measures, with regard to Article 2, point
(6) of this Directive.
2. To that end, this Directive lays down:
(a) obligations on Member States;
(b) risk-management measures for entities:
(i) essential entities;
(ii) important entities;
(c) rules on information sharing.
Article 2
Definitions
For the purposes of this Directive, the following definitions apply:
(1) \u2018incident\u2019 means an event compromising the availability of data;
(2) \u2018vulnerability\u2019 means a weakness of ICT products or ICT services that can be
exploited by a cyber threat;
(3) Directive 2005/29/EC of the European Parliament (OJ L 149, 11.6.2005, p. 22).
""" + FOOTER

PAGE_3 = """Article 3
Measures
1. The measures shall include at least the following:
(a) basic cyber hygiene practices;
(b) policies regarding the use of cryptography;
(c) human resources security, access control policies and asset rechar\u00ad
ging management.
Done at Strasbourg, 14 December 2022.
ANNEX I
SECTORS OF HIGH CRITICALITY
Sector Subsector Type of entity
1. Energy (a) Electricity — Electricity undertakings
(b) Oil — Operators of oil transmission pipelines
2. Digital providers — Providers of online marketplaces
"""


def _pages(*texts: str) -> list[Document]:
    return [
        Document(page_content=text, metadata={"source": "act.pdf", "page": n, "total_pages": 3})
        for n, text in enumerate(texts, start=1)
    ]


@pytest.fixture
def chunks():
    return split_structured(_pages(PAGE_1, PAGE_2, PAGE_3))[0]


def _by_ref(chunks):
    return {c.metadata["ref"]: c for c in chunks}


def test_footers_and_footnotes_are_dropped():
    text = "\n".join(line.text for line in document_lines(_pages(PAGE_1, PAGE_2)))
    assert "Official Journal of the European Union L 333" not in text
    assert "OJ C 233" not in text
    # "(3) Directive 2005/29/EC" sits exactly where definition (3) is expected.
    assert "2005/29/EC" not in text
    assert "(4) aimed" in text  # an inline footnote *reference* is body text


def test_one_unit_per_recital_paragraph_and_definition(chunks):
    refs = [c.metadata["ref"] for c in chunks]
    assert refs == [
        "Preamble", "Recital 1", "Recital 2",
        "Art. 1(1)", "Art. 1(2)",
        "Art. 2", "Art. 2, point (1)", "Art. 2, point (2)",
        "Art. 3(1)", "Signature",
        "Annex I, point 1(a)", "Annex I, point 1(b)", "Annex I, point 2",
    ]


def test_wrapped_lines_do_not_open_units(chunks):
    by_ref = _by_ref(chunks)
    assert "(TFEU), the objective" in by_ref["Recital 2"].page_content
    assert "(6) of this Directive." in by_ref["Art. 1(1)"].page_content


def test_breadcrumb_is_embedded_with_the_text(chunks):
    by_ref = _by_ref(chunks)
    assert by_ref["Art. 1(2)"].page_content.startswith(
        "Article 1 — Subject matter, paragraph 2\n"
    )
    assert by_ref["Art. 2, point (2)"].page_content.startswith(
        "Article 2 — Definitions, point (2)\n"
    )
    assert by_ref["Annex I, point 1(b)"].page_content.startswith(
        "Annex I — Sectors of high criticality, point 1 (Energy), subsector (b)\n"
    )


def test_metadata(chunks):
    meta = _by_ref(chunks)["Art. 1(2)"].metadata
    assert meta["section"] == "article"
    assert (meta["article"], meta["paragraph"], meta["article_title"]) == (1, 2, "Subject matter")
    assert meta["page"] == 2 and meta["pages"] == [2]
    assert _by_ref(chunks)["Recital 2"].metadata["recital"] == 2
    assert _by_ref(chunks)["Art. 2, point (2)"].metadata["definition"] == 2
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_points_nest_roman_under_a_letter_that_opens_a_list():
    units = {u.ref: u for u in parse(document_lines(_pages(PAGE_1, PAGE_2)))}
    points = units["Art. 1(2)"].points
    assert [p[0].text[:3] for p in points] == ["(a)", "(b)", "(c)"]
    assert [line.text[:4] for line in points[1]] == ["(b) ", "(i) ", "(ii)"]


def test_i_after_h_is_the_ninth_letter():
    # Article 21(2): ten measures (a)-(j), where "(i)" follows "(h) ...;".
    items = [f"({c}) measure {c};" for c in "abcdefghij"]
    page = "\n".join(["Article 1", "Measures", "1. Entities shall take:", *items, "Article 2", "X"])
    units = {u.ref: u for u in parse(document_lines(_pages(page)))}
    assert [p[0].text[:3] for p in units["Art. 1(1)"].points] == [f"({c})" for c in "abcdefghij"]


def test_a_list_must_start_at_a():
    page = "Article 1\nScope\n1. It covers:\n(g) a wrapped reference;\nArticle 2\nX\nY."
    units = {u.ref: u for u in parse(document_lines(_pages(page)))}
    assert units["Art. 1(1)"].points == []


def test_soft_hyphen_rejoins_a_word_split_across_lines(chunks):
    assert "recharging management" in _by_ref(chunks)["Art. 3(1)"].page_content


def test_unit_spanning_a_page_break_lists_both_pages():
    page_a = "Article 1\nScope\n1. This Directive applies to entities\n"
    page_b = "that provide services in the Union.\nArticle 2\nOther\nText."
    chunks, _ = split_structured(_pages(page_a, page_b))
    first = _by_ref(chunks)["Art. 1(1)"]
    assert first.metadata["pages"] == [1, 2] and first.metadata["page"] == 1


def test_oversized_unit_splits_between_points_repeating_the_lead():
    lead = "1. Entities shall submit the following:"
    items = [f"({c}) {'report item ' * 12}{c};" for c in "abcdef"]
    page = "\n".join(["Article 1", "Reporting", lead, *items, "Article 2", "Other", "Text."])
    unit = {u.ref: u for u in parse(document_lines(_pages(page)))}["Art. 1(1)"]
    parts = pieces(unit, max_chars=400)
    assert len(parts) > 1
    for text, _ in parts:
        assert text.startswith(lead) and len(text) <= 400
    body = "\n".join(text for text, _ in parts)
    assert all(item in body for item in items)  # nothing lost across the cut


def test_ids_are_unique_and_stable():
    first = split_structured(_pages(PAGE_1, PAGE_2, PAGE_3))[1]
    second = split_structured(_pages(PAGE_1, PAGE_2, PAGE_3))[1]
    assert first == second and len(set(first)) == len(first)


def test_pdf_without_articles_falls_back_to_fixed_chunks():
    page = "Quarterly report\n" + "Revenue grew in every region. " * 60
    chunks, ids = split_structured(_pages(page))
    assert chunks and len(chunks) == len(ids)
    assert "section" not in chunks[0].metadata


def test_factory_tags_chunks_and_rejects_unknown_chunker():
    chunks, _ = split(_pages(PAGE_1, PAGE_2, PAGE_3), chunker="structured")
    assert {c.metadata["chunker"] for c in chunks} == {"structured"}
    with pytest.raises(ValueError, match="Unknown chunker"):
        split(_pages(PAGE_1), chunker="nope")


def test_retrieval_metrics_credit_every_page_a_chunk_spans():
    entry = {"evidence": [{"source": "a.pdf", "page": 45}]}
    assert retrieval_metrics(entry, [("a.pdf", [44, 45])])["rank"] == 1
    assert retrieval_metrics(entry, [("a.pdf", 44)])["hit"] is False


NIS2 = RAW_PDF_DIR / "CELEX_32022L2555_EN_TXT.pdf"


@pytest.mark.skipif(not NIS2.exists(), reason="NIS2 PDF not in data/raw_pdfs/")
def test_real_directive_structure():
    from app.ingestion.loader import load_pdf

    pages = load_pdf(NIS2)
    chunks, ids = split_structured(pages)
    meta = [c.metadata for c in chunks]

    assert max(m.get("recital", 0) for m in meta) == 144
    assert {m["article"] for m in meta if "article" in m} == set(range(1, 47))
    assert max(m.get("definition", 0) for m in meta) == 41
    assert len(set(ids)) == len(ids)
    # No footnote survives: they are the only text citing the Official Journal.
    assert not [c.metadata["ref"] for c in chunks if "(OJ " in c.page_content]
    # The steps' known misses now each have a unit of their own.
    by_ref = _by_ref(chunks)
    assert "\u2018vulnerability\u2019 means" in by_ref["Art. 6, point (15)"].page_content
    assert "online marketplaces" in by_ref["Annex II, point 6"].page_content
    assert "By 17 October 2024" in by_ref["Art. 41(1)"].page_content
    measures = by_ref["Art. 21(2)"].page_content  # all ten measures, one chunk
    assert "(a) policies on risk analysis" in measures and "(j) the use of multi-factor" in measures


def test_corpus_guard_refuses_rechunking_in_place(monkeypatch):
    import app.ingestion.pipeline as pipeline

    asked = {}

    def fake_indexed(sources):
        asked["sources"] = sources
        return "fixed"

    monkeypatch.setattr(pipeline, "indexed_chunker", fake_indexed)
    with pytest.raises(ValueError, match="different QDRANT_COLLECTION"):
        pipeline.check_chunker("structured", ["nis2.pdf"])
    assert asked["sources"] == ["nis2.pdf"]  # only the corpus is checked, not uploads
    pipeline.check_chunker("fixed", ["nis2.pdf"])  # same chunker: fine
