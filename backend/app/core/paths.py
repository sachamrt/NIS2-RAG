"""Filesystem anchors.

Paths are resolved from this file's location, not the working directory, so
`pytest`, `uvicorn` and the CLI scripts behave the same no matter where they
are launched from.
"""

from pathlib import Path

# backend/app/core/paths.py -> core -> app -> backend -> repo root
BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent

ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"

# The corpus you curate by hand, and the only PDFs kept on disk. Uploads are
# ingested from a temp file and never stored: once the chunks are in Qdrant the
# PDF has no further use, and a rebuild is expected to lose them.
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
