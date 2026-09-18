"""Smoke-check the configured providers end to end.

    python scripts/check_providers.py            # use .env
    python scripts/check_providers.py --llm openai --embeddings local
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root

from app.core.config import get_settings
from app.core.embeddings_factory import build_embeddings
from app.core.llm_factory import build_llm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", help="override LLM_PROVIDER for this run")
    parser.add_argument("--embeddings", help="override EMBEDDING_PROVIDER for this run")
    args = parser.parse_args()

    settings = get_settings()
    llm_provider = args.llm or settings.llm_provider
    emb_provider = args.embeddings or settings.embedding_provider

    llm = build_llm(llm_provider, settings)
    reply = llm.invoke("Reply with the single word: ok")
    print(f"llm        {llm_provider:<10} {type(llm).__name__} -> {reply.content.strip()[:40]!r}")

    embeddings = build_embeddings(emb_provider, settings)
    vector = embeddings.embed_query("NIS2 directive")
    print(f"embeddings {emb_provider:<10} {type(embeddings).__name__} -> dim {len(vector)}")
    print(f"\nQdrant collection {settings.qdrant_collection!r} must hold {len(vector)}-dim vectors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
