"""Ingest PDFs from data/raw_pdfs into Qdrant.

    python scripts/ingest_cli.py
    python scripts/ingest_cli.py --dir data/raw_pdfs --chunk-size 800
    python scripts/ingest_cli.py --stats        # just show what's in the collection
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root

from app.core.config import get_settings
from app.core.vectorstore import clear, collection_stats
from app.ingestion.loader import RAW_PDF_DIR, discover_pdfs
from app.ingestion.pipeline import BATCH_SIZE, ingest_directory
from app.ingestion.splitter import CHUNK_OVERLAP, CHUNK_SIZE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(RAW_PDF_DIR), help="directory of PDFs")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--stats", action="store_true", help="show collection stats and exit")
    parser.add_argument(
        "--clear",
        metavar="SOURCE",
        nargs="?",
        const="*",
        help="delete a source's chunks (or all, if given no value) and exit",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.stats:
        print(collection_stats())
        return 0

    if args.clear:
        removed = clear(None if args.clear == "*" else args.clear)
        print(f"deleted chunks for {removed}")
        print(collection_stats())
        return 0

    pdfs = discover_pdfs(args.dir)
    if not pdfs:
        print(f"No PDFs found in {args.dir}/ -- drop the NIS2 directive there first.")
        return 1

    print(f"embedder {settings.embedding_provider}/{settings.mistral_embedding_model}")
    print(f"collection {settings.qdrant_collection}")
    print(f"{len(pdfs)} PDF(s) to ingest\n")

    # Only animate on a real terminal; piped/CI output stays line-per-event.
    tty = sys.stdout.isatty()

    def progress(path: Path, done: int, total: int) -> None:
        end = "\r" if tty else "\n"
        print(f"  {path.name}: {done}/{total} chunks".ljust(60), end=end, flush=True)

    report = ingest_directory(
        args.dir, args.chunk_size, args.chunk_overlap, args.batch_size, progress
    )
    if tty:
        print(" " * 60, end="\r")
    for record in report.per_file:
        print(f"  {record['file']}: {record['pages']} pages -> {record['chunks']} chunks")
    print(f"\n{report.summary()}")
    print(collection_stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
