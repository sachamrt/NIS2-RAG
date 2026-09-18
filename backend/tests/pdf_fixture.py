"""Minimal valid PDF writer, so ingestion tests need no binary fixtures."""

from pathlib import Path


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def write_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Write a PDF where `pages` is a list of pages, each a list of text lines."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # 1-based object number

    catalog_num = 1
    pages_num = 2
    objects.extend([b"", b""])  # reserved slots for catalog and page tree

    font_num = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    kids: list[int] = []
    for lines in pages:
        ops = ["BT", "/F1 12 Tf", "50 750 Td", "14 TL"]
        ops += [f"({_escape(line)}) Tj T*" for line in lines]
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content_num = add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        page_num = add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
            b"/Contents %d 0 R /Resources << /Font << /F1 %d 0 R >> >> >>"
            % (pages_num, content_num, font_num)
        )
        kids.append(page_num)

    objects[catalog_num - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_num
    objects[pages_num - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids),
        len(kids),
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (num, body)

    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        catalog_num,
        xref_at,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path
