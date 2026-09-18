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
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
