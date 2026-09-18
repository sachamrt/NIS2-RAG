"""Check the configured LLM only -- no embeddings, one API call.

    python scripts/check_llm.py
    python scripts/check_llm.py --provider openai
    python scripts/check_llm.py --model mistral-small-latest
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from repo root

from app.core.config import get_settings
from app.core.llm_factory import build_llm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="override LLM_PROVIDER for this run")
    parser.add_argument("--model", help="override the provider's model for this run")
    parser.add_argument("--prompt", default="Reply with the single word: ok")
    args = parser.parse_args()

    settings = get_settings()
    provider = args.provider or settings.llm_provider
    if args.model:
        settings = settings.model_copy(update={f"{provider}_llm_model": args.model})

    llm = build_llm(provider, settings)
    model = getattr(llm, "model", getattr(llm, "model_name", "?"))
    print(f"provider {provider} | model {model}")

    try:
        reply = llm.invoke(args.prompt)
    except Exception as exc:
        print(f"FAIL  {type(exc).__name__}: {exc}"[:400])
        return 1

    print(f"OK    {reply.content.strip()[:80]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
