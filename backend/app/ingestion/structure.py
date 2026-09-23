"""Structure-aware chunking for EU legal acts (directives, regulations).

The fixed splitter cuts every 1000 characters, blind to the text. This one cuts
where the act itself does: one chunk per recital, per numbered Article
paragraph, per definition, per Annex row. Each chunk starts with a breadcrumb
("Article 21 — Cybersecurity risk-management measures, paragraph 2") that is
embedded with it, because a paragraph rarely names its own Article's subject.

Parsing is one pass over the document's *lines*, not its pages -- a paragraph
can straddle a page break -- driven by a small state machine. The same marker
means different things by context: "(4)" opens a recital in the preamble, a
definition inside Article 6, and is a footnote at the foot of a page. A marker
is only accepted if it is the *next expected* one (recital 5 after recital 4,
point (c) after (b)); that is what rejects line wraps like "(TFEU), the ...".

Used for the curated corpus only; uploads always take the fixed splitter. A
corpus PDF with no Article structure (not an EU act) falls back to the fixed
splitter too, so nothing is ever left unindexed.
"""

import hashlib
import re
import uuid
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.splitter import SEPARATORS, split_documents

# Longest chunk body before a unit is split. Article 21(2) -- the ten measures,
# which must reach the model as one list -- is ~1400 characters.
MAX_CHARS = 1600
FALLBACK_OVERLAP = 150

# Fewer Article headings than this and the PDF is not treated as a legal act.
MIN_ARTICLES = 2

_NAMESPACE = uuid.UUID("0c6a8f52-9e41-5d7b-8a3f-2b1c4d5e6f70")

# "EN Official Journal of the European Union L 333/81 27.12.2022", both orders.
_FOOTER = re.compile(
    r"^[A-Z]{2} Official Journal of the European Union"
    r"(?: +(?:[LC] \d+/\d+|\d{1,2}\.\d{1,2}\.\d{4})){2}$"
)
_CITES_OJ = re.compile(r"\bOJ [LC]\b|Official Journal")

_WHEREAS = re.compile(r"^Whereas:?$")
_ENACTING = re.compile(r"^HA(?:VE|S) ADOPTED THIS [A-Z]+:?$")
_DIVISION = re.compile(r"^(?:CHAPTER|TITLE|SECTION) [IVXLC\d]+$")
_ARTICLE = re.compile(r"^Article (\d+)$")
_ANNEX = re.compile(r"^ANNEX(?: ([IVXLC]+))?$")
_CLOSING = re.compile(r"^Done at ")

_NUMBERED = re.compile(r"^\((\d+)\) ")  # recital, definition or footnote
# (15) 'vulnerability' means ... -- opens with a curly or straight quote
_DEFINITION = re.compile("^\\((\\d+)\\) [\u2018\u201c'\"]")
_PARAGRAPH = re.compile(r"^(\d+)\. ")  # 2. The measures ...  /  6. Digital providers
_POINT = re.compile(r"^\(([a-z]{1,5})\) ")  # (a) ...  /  (iv) ...
_INLINE_SUBSECTOR = re.compile(r"^\d+\. (.{1,60}?) \(a\) ")  # 1. Energy (a) Electricity
# A list item ends like this; a line wrapped mid-sentence does not. Guards
# "... referred to in point\n(d) of paragraph 1" from opening point (d).
_ITEM_END = re.compile(r"(?:[;:,.]|\b(?:and|or))$")

_ROMAN = "i ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi xvii xviii xix xx".split()


@dataclass
class Line:
    text: str
    page: int


@dataclass
class Unit:
    """One structural unit: a recital, an Article paragraph, a definition, ...

    ``lead`` is the text before the first point, ``points`` one line-list per
    point (a), (b), ... with its sub-points (i), (ii) folded in. Oversized
    units split between points, repeating the lead in every piece.
    """

    section: str  # preamble | recital | article | annex | closing
    heading: str
    ref: str
    meta: dict = field(default_factory=dict)
    lead: list[Line] = field(default_factory=list)
    points: list[list[Line]] = field(default_factory=list)
    next_letter: str = "a"
    next_roman: str = "i"
    in_roman: bool = False

    @property
    def lines(self) -> list[Line]:
        return self.lead + [line for point in self.points for line in point]

    def add(self, line: Line) -> None:
        (self.points[-1] if self.points else self.lead).append(line)

    def last_text(self) -> str:
        lines = self.lines
        return lines[-1].text if lines else ""

    def try_point(self, mark: str, line: Line) -> bool:
        """Open point `mark` if it is the next one here; False means continuation."""
        prev = self.last_text()
        if not _ITEM_END.search(prev):
            return False
        # Roman first: after "(h) ... the following:" an "(i)" is a sub-point,
        # while after "(h) ...;" it is the ninth letter (Article 21(2)).
        if self.points and mark == self.next_roman and (self.in_roman or prev.endswith(":")):
            self.points[-1].append(line)
            self.in_roman = True
            self.next_roman = _after(_ROMAN, mark)
            return True
        if mark == self.next_letter and (self.points or prev.endswith(":")):
            self.points.append([line])
            self.next_letter = chr(ord(mark) + 1) if len(mark) == 1 else ""
            self.next_roman, self.in_roman = "i", False
            return True
        return False


def _after(sequence: list[str], item: str) -> str:
    i = sequence.index(item)
    return sequence[i + 1] if i + 1 < len(sequence) else ""


# --- page cleaning ---------------------------------------------------------


def _footnote_start(lines: list[str]) -> int:
    """Index where the page's footnote block begins, or len(lines) if none.

    Footnotes are numbered "(N)" like recitals and definitions, and sit at the
    foot of the page. The block is the earliest numbered line from which every
    numbered entry to the bottom is consecutive *and* cites the Official
    Journal. Numbering restarts in the Annexes, so no running counter is used.
    """
    for i, line in enumerate(lines):
        if _NUMBERED.match(line) and _is_footnote_block(lines[i:]):
            return i
    return len(lines)


def _is_footnote_block(lines: list[str]) -> bool:
    entries: list[list] = []
    for line in lines:
        if m := _NUMBERED.match(line):
            entries.append([int(m[1]), line])
        else:
            entries[-1][1] += " " + line
    first = entries[0][0]
    return [n for n, _ in entries] == list(range(first, first + len(entries))) and all(
        _CITES_OJ.search(text) for _, text in entries
    )


def document_lines(pages: list[Document]) -> list[Line]:
    """Every content line of the document, in order, tagged with its page.

    Drops the running footer and the footnotes; rejoins words hyphenated across
    a line break with a soft hyphen ("rechar\\xadging" -> "recharging").
    """
    lines: list[Line] = []
    for doc in pages:
        page = int(doc.metadata.get("page", 0))
        raw = [text.strip() for text in doc.page_content.splitlines()]
        raw = [text for text in raw if text and not _FOOTER.match(text)]
        for text in raw[: _footnote_start(raw)]:
            if lines and lines[-1].text.endswith("\u00ad"):
                lines[-1].text = lines[-1].text[:-1] + text
            else:
                lines.append(Line(text, page))
    for line in lines:
        line.text = line.text.replace("\u00ad", "")
    return lines


# --- parsing ---------------------------------------------------------------


class _Parser:
    def __init__(self) -> None:
        self.units: list[Unit] = []
        self.section = "preamble"
        self.unit = self._open(Unit("preamble", "Preamble", "Preamble"))
        self.expect: str | None = None  # a title line is due next
        self.next_recital = 1
        self.next_article = 1
        # per article
        self.article: int | None = None
        self.article_title = ""
        self.next_paragraph = 1
        self.next_definition = 1
        # per annex
        self.annex = ""
        self.annex_title = ""
        self.annex_header = ""
        self.next_row = 1
        self.row: int | None = None
        self.sector = ""

    def _open(self, unit: Unit) -> Unit:
        self.units.append(unit)
        self.unit = unit
        return unit

    def feed(self, line: Line) -> None:
        text = line.text

        if self.expect == "division_title":  # "GENERAL PROVISIONS" under "CHAPTER I"
            self.expect = None
            return
        if self.expect == "article_title":
            self.expect = None
            self.article_title = text
            self.unit.heading = f"Article {self.article} — {text}"
            self.unit.meta["article_title"] = text
            return

        if (m := _ANNEX.match(text)) and self.section in ("article", "closing", "annex"):
            self._open_annex(m[1] or "")
            return
        if self.section == "annex":
            self._annex_line(line)
            return

        if _CLOSING.match(text) and self.section == "article":
            self.section = "closing"
            self._open(Unit("closing", "Signature", "Signature")).add(line)
            return
        if self.section == "closing":
            self.unit.add(line)
            return

        if _ENACTING.match(text):
            self.section = "article"
            return
        if _DIVISION.match(text) and self.section == "article":
            self.expect = "division_title"
            return
        # Recitals end only at the enacting formula: an "Article 5" line
        # inside a recital is a wrapped reference, not a heading.
        m = _ARTICLE.match(text)
        if m and self.section != "recital" and int(m[1]) == self.next_article:
            self._open_article(int(m[1]))
            return

        if self.section == "preamble":
            self.unit.add(line)
            if _WHEREAS.match(text):
                self.section = "recital"
        elif self.section == "recital":
            m = _NUMBERED.match(text)
            if m and int(m[1]) == self.next_recital:
                n = self.next_recital
                self.next_recital += 1
                self._open(Unit("recital", f"Recital {n}", f"Recital {n}", {"recital": n}))
            self.unit.add(line)
        else:
            self._article_line(line)

    def _open_article(self, n: int) -> None:
        self.section = "article"
        self.article, self.article_title = n, ""
        self.next_article = n + 1
        self.next_paragraph = self.next_definition = 1
        self.expect = "article_title"
        self._open(Unit("article", f"Article {n}", f"Art. {n}", {"article": n}))

    def _article_line(self, line: Line) -> None:
        text = line.text
        base = f"Article {self.article} — {self.article_title}"
        meta = {"article": self.article, "article_title": self.article_title}

        if (m := _PARAGRAPH.match(text)) and int(m[1]) == self.next_paragraph:
            n = self.next_paragraph
            self.next_paragraph += 1
            unit = Unit(
                "article", f"{base}, paragraph {n}", f"Art. {self.article}({n})",
                {**meta, "paragraph": n},
            )
            self._open(unit).add(line)
        elif (m := _DEFINITION.match(text)) and int(m[1]) == self.next_definition:
            n = self.next_definition
            self.next_definition += 1
            unit = Unit(
                "article", f"{base}, point ({n})", f"Art. {self.article}, point ({n})",
                {**meta, "definition": n},
            )
            self._open(unit).add(line)
        elif (m := _POINT.match(text)) and self.unit.try_point(m[1], line):
            pass
        else:
            self.unit.add(line)

    def _open_annex(self, number: str) -> None:
        self.section = "annex"
        self.annex, self.annex_title, self.annex_header = number, "", ""
        self.next_row, self.row = 1, None
        self.expect = "annex_title"
        name = f"Annex {number}".strip()
        self._open(Unit("annex", name, name, {"annex": number}))

    def _annex_line(self, line: Line) -> None:
        text = line.text
        name = f"Annex {self.annex}".strip()

        if self.expect == "annex_title":
            self.expect = "annex_header"
            self.annex_title = text.capitalize()
            self.unit.heading = f"{name} — {self.annex_title}"
            return
        if self.expect == "annex_header":
            # Tables repeat their column header on every page; keep the first.
            self.expect = None
            self.annex_header = text
            self.unit.add(line)
            return
        if text == self.annex_header:
            return

        heading = f"{name} — {self.annex_title}"
        meta = {"annex": self.annex}
        if (m := _PARAGRAPH.match(text)) and int(m[1]) == self.next_row:
            n = self.row = self.next_row
            self.next_row += 1
            if self.unit.ref == name and [x.text for x in self.unit.lines] == [self.annex_header]:
                self.units.pop()  # the intro held nothing but the column header
            inline = _INLINE_SUBSECTOR.match(text)
            self.sector = inline[1] if inline and "—" not in inline[1] else ""
            if self.sector:  # "1. Energy (a) Electricity — ..." opens subsector (a)
                unit = Unit(
                    "annex", f"{heading}, point {n} ({self.sector}), subsector (a)",
                    f"{name}, point {n}(a)", {**meta, "annex_point": n, "annex_subpoint": "a"},
                    next_letter="b",
                )
            else:
                unit = Unit("annex", f"{heading}, point {n}", f"{name}, point {n}",
                            {**meta, "annex_point": n})
            self._open(unit).add(line)
            return

        m = _POINT.match(text)
        if m and self.row is not None and m[1] == self.unit.next_letter and len(m[1]) == 1:
            sub, n = m[1], self.row
            label = f"point {n} ({self.sector})" if self.sector else f"point {n}"
            unit = Unit(
                "annex", f"{heading}, {label}, subsector ({sub})", f"{name}, point {n}({sub})",
                {**meta, "annex_point": n, "annex_subpoint": sub},
                next_letter=chr(ord(sub) + 1),
            )
            self._open(unit).add(line)
            return
        self.unit.add(line)


def parse(lines: list[Line]) -> list[Unit]:
    """Group the document's lines into structural units, in document order."""
    parser = _Parser()
    for line in lines:
        parser.feed(line)
    return [unit for unit in parser.units if unit.lines]


# --- chunking --------------------------------------------------------------


def _join(lines: list[Line]) -> str:
    return "\n".join(line.text for line in lines)


def _pages(lines: list[Line]) -> list[int]:
    return sorted({line.page for line in lines})


def _split_long(lines: list[Line], max_chars: int) -> list[tuple[str, list[int]]]:
    """Last resort for a unit with no points to split on (a long recital)."""
    body = _join(lines)
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line.text) + 1
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=FALLBACK_OVERLAP,
        separators=SEPARATORS,
        add_start_index=True,
    )
    pieces = []
    for doc in splitter.create_documents([body]):
        begin = doc.metadata["start_index"]
        end = begin + len(doc.page_content)
        pages = {
            line.page
            for line, s in zip(lines, starts, strict=True)
            if s < end and s + len(line.text) > begin
        }
        pieces.append((doc.page_content, sorted(pages)))
    return pieces


def pieces(unit: Unit, max_chars: int = MAX_CHARS) -> list[tuple[str, list[int]]]:
    """A unit's body as one or more (text, pages) pieces of at most max_chars.

    Splits between points and repeats the lead in each piece: without "...
    shall include at least the following:", a piece starting at point (f)
    would not say these are obligations.
    """
    lines = unit.lines
    if len(_join(lines)) <= max_chars:
        return [(_join(lines), _pages(lines))]

    groups: list[list[Line]] = [lines]
    if unit.points:
        groups, current = [], []
        for point in unit.points:
            candidate = current + point
            if current and len(_join(unit.lead + candidate)) > max_chars:
                groups.append(unit.lead + current)
                candidate = point
            current = candidate
        groups.append(unit.lead + current)

    out: list[tuple[str, list[int]]] = []
    for group in groups:
        if len(_join(group)) <= max_chars:
            out.append((_join(group), _pages(group)))
        else:
            out.extend(_split_long(group, max_chars))
    return out


def chunk_id(source: str, ref: str, part: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return str(uuid.uuid5(_NAMESPACE, f"{source}:{ref}:{part}:{digest}"))


def split_structured(
    pages: list[Document], max_chars: int = MAX_CHARS
) -> tuple[list[Document], list[str]]:
    """Page Documents of one PDF -> (chunks, ids), one chunk per structural unit.

    Metadata keeps ``page`` (first page) so citations and the eval work as
    before, and adds ``pages``, ``section``, ``ref`` ("Art. 21(2)") and the
    unit's numbers (``article``, ``paragraph``, ``recital``, ...).
    """
    if not pages:
        return [], []
    units = parse(document_lines(pages))
    if len({u.meta["article"] for u in units if u.section == "article"}) < MIN_ARTICLES:
        return split_documents(pages)

    base = {k: v for k, v in pages[0].metadata.items() if k in ("source", "total_pages")}
    source = base.get("source", "unknown")
    chunks: list[Document] = []
    ids: list[str] = []
    for unit in units:
        parts = pieces(unit, max_chars)
        for part, (text, unit_pages) in enumerate(parts):
            content = f"{unit.heading}\n{text}"
            metadata = {
                **base,
                "page": unit_pages[0],
                "pages": unit_pages,
                "section": unit.section,
                "ref": unit.ref,
                **{k: v for k, v in unit.meta.items() if v is not None},
                "chunk_index": len(chunks),
            }
            if len(parts) > 1:
                metadata["part"] = part + 1
            chunks.append(Document(page_content=content, metadata=metadata))
            ids.append(chunk_id(source, unit.ref, part, content))
    return chunks, ids
