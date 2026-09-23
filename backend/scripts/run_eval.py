"""Run the evaluation set against the RAG pipeline.

python scripts/run_eval.py --retrieval-only          # free: embeddings only, no LLM
python scripts/run_eval.py                           # + answers, phrase checks
python scripts/run_eval.py --judge                   # + LLM judge (correctness, faithfulness)
python scripts/run_eval.py --judge --save-baseline   # record the reference run
python scripts/run_eval.py --judge --compare         # diff against the baseline

--type lookup | --only id1,id2 | --limit 5 | --k 6 | --judge-model ministral-14b-latest
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from app.core.config import get_settings
from app.core.llm_factory import build_llm
from app.core.paths import BACKEND_ROOT
from app.core.vectorstore import collection_stats
from app.evaluation.judge import JudgeError, judge
from app.evaluation.metrics import (
    EVAL_FILE,
    load_dataset,
    phrase_checks,
    retrieval_metrics,
    summarize,
)
from app.ingestion.splitter import CHUNK_OVERLAP, CHUNK_SIZE
from app.rag.chain import answer_with_hits, retrieve
from app.rag.prompts import format_context

RESULTS_DIR = BACKEND_ROOT / "eval" / "results"
BASELINE = BACKEND_ROOT / "eval" / "baseline.json"

# A run is only comparable to another if these match. The first group is what
# you tune -- differences there are the point. The second group must be equal,
# or the comparison measures a change of ruler rather than a change of system.
TUNABLE = ("llm_model", "embedding_model", "k", "chunk_size", "chunk_overlap", "points")
MUST_MATCH = ("dataset_sha", "judge_model")


# --- config ----------------------------------------------------------------


def fingerprint(args, settings) -> dict:
    llm = getattr(settings, f"{settings.llm_provider}_llm_model", settings.llm_provider)
    emb = getattr(settings, f"{settings.embedding_provider}_embedding_model", "?")
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": llm,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": emb,
        "temperature": settings.llm_temperature,
        "k": args.k or settings.retrieval_k,
        # The splitter constants describe how the index *should* have been
        # built; `points` fingerprints how it actually was.
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "collection": settings.qdrant_collection,
        "points": collection_stats()["points"],
        "dataset_sha": hashlib.sha256(EVAL_FILE.read_bytes()).hexdigest()[:12],
        "judge_model": (args.judge_model or llm) if args.judge else None,
        "mode": "retrieval-only"
        if args.retrieval_only
        else ("judged" if args.judge else "answers"),
    }


def build_judge(args, settings):
    provider = args.judge_provider or settings.llm_provider
    overrides = {"llm_temperature": 0.0}
    if args.judge_model:
        overrides[f"{provider}_llm_model"] = args.judge_model
    return build_llm(provider, settings.model_copy(update=overrides))


# --- running ---------------------------------------------------------------


def with_retry(fn, attempts: int = 4, base_delay: float = 8.0):
    """Retry on rate limits only. Anything else is a real failure: re-raise."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            limited = "429" in str(exc) or "rate" in str(exc).lower()
            if not limited or attempt == attempts - 1:
                raise
            wait = base_delay * 2**attempt
            print(f"      rate-limited, retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def run_entry(entry: dict, args, judge_llm) -> dict:
    result = {"id": entry["id"], "type": entry["type"], "question": entry["question"]}
    start = time.perf_counter()
    try:
        if args.retrieval_only:
            hits = with_retry(lambda: retrieve(entry["question"], args.k))
            text = None
        else:
            text, hits = with_retry(lambda: answer_with_hits(entry["question"], k=args.k))

        retrieved = [
            (d.metadata.get("source"), d.metadata.get("pages") or int(d.metadata.get("page", 0)))
            for d, _ in hits
        ]
        result["retrieved"] = [
            {"source": d.metadata.get("source"), "page": int(d.metadata.get("page", 0)),
             "ref": d.metadata.get("ref")}
            for d, _ in hits
        ]
        result["retrieval"] = retrieval_metrics(entry, retrieved)

        if text is not None:
            result["answer"] = text
            result["checks"] = phrase_checks(entry, text)

        if judge_llm is not None and text is not None:
            context = format_context([d for d, _ in hits])
            try:
                result["judge"] = with_retry(
                    lambda: judge(
                        judge_llm,
                        question=entry["question"],
                        reference=entry["reference_answer"],
                        context=context,
                        answer=text,
                    )
                )
            except JudgeError as exc:
                # Unparseable verdict: keep the entry, exclude it from judge stats.
                result["judge"] = None
                result["judge_error"] = str(exc)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:300]
        result.setdefault("retrieval", {"hit": None, "rank": None, "rr": None, "page_recall": None})

    result["seconds"] = round(time.perf_counter() - start, 2)
    return result


def progress_line(i: int, n: int, r: dict) -> str:
    if r.get("error"):
        return f"[{i:>2}/{n}] ERROR {r['id']}  {r['error'][:70]}"
    ret = r["retrieval"]
    if ret["hit"] is None:
        rank = "  -- "
    elif ret["hit"]:
        rank = f"hit@{ret['rank']}"
    else:
        rank = "MISS "
    parts = [f"[{i:>2}/{n}]", f"{rank:5}"]
    if r.get("checks"):
        c = r["checks"]
        if r["type"] == "unanswerable":
            parts.append("refused" if c["refused"] else "ANSWERED")
        else:
            parts.append(
                {True: "inc ok ", False: "inc MISS", None: "       "}[c["must_include_ok"]]
            )
        if c["hallucination_flags"]:
            parts.append(f"HALLUC {c['hallucination_flags']}")
    if r.get("judge"):
        j = r["judge"]
        parts.append(f"{j['correctness']:9} {'faithful' if j['faithful'] else 'UNFAITHFUL'}")
    elif r.get("judge_error"):
        parts.append("judge?")
    parts.append(f"{r['id']}  ({r['seconds']}s)")
    return "  ".join(parts)


# --- reporting -------------------------------------------------------------


def pct(v) -> str:
    return "  n/a" if v is None else f"{v * 100:5.1f}%"


def print_summary(s: dict) -> None:
    r, g = s["retrieval"], s["generation"]
    print(f"\n{'=' * 64}\n{s['entries']} entries, {s['errors']} errors\n")
    print(f"Retrieval  (n={r['n']})")
    print(f"  hit rate      {pct(r['hit_rate'])}   an expected page was retrieved")
    mrr = "n/a" if r["mrr"] is None else r["mrr"]
    print(f"  MRR           {mrr:>6}   1.0 = always ranked first")
    print(f"  page recall   {pct(r['page_recall'])}   share of expected pages found")
    if g["n"]:
        print(f"\nAnswers, phrase checks  (n={g['n']})")
        print(f"  must_include        {pct(g['must_include_rate'])}")
        print(f"  refused unanswerable{pct(g['refusal_rate_unanswerable'])}")
        print(f"  hallucination flags {g['hallucination_flags']:>6}")
    if "judge" in s:
        j = s["judge"]
        print(f"\nJudge  (n={j['n']})")
        print(f"  correct       {pct(j['correct_rate'])}")
        print(f"  partial       {pct(j['partial_rate'])}")
        print(f"  faithful      {pct(j['faithful_rate'])}   every claim supported by the context")
        print(f"  unanswerable  {pct(j['unanswerable_correct'])}   handled without inventing facts")

    if s.get("disagreements"):
        print(f"\nPhrase check and judge disagree -- read these ({len(s['disagreements'])}):")
        for d in s["disagreements"]:
            label = "flagged" if d["phrases"] == "halluc" else "missing"
            extra = f"  {label} {d['missing']}" if d.get("missing") else ""
            print(f"  {d['id']:38} phrases {d['phrases']:4}  judge {d['judge']}{extra}")

    print(f"\n{'type':13} {'n':>3}  {'hit':>6}  {'incl':>6}  {'correct':>7}")
    for t, v in s["by_type"].items():
        print(
            f"{t:13} {v['n']:>3}  {pct(v['hit_rate'])}  {pct(v['must_include_rate'])}  "
            f"{pct(v.get('correct_rate'))}"
        )


def compare(current: dict, baseline: dict) -> None:
    cfg_now, cfg_base = current["config"], baseline["config"]
    print(f"\n{'=' * 64}\nCompared with baseline from {baseline['timestamp']}")

    broken = [k for k in MUST_MATCH if cfg_now.get(k) != cfg_base.get(k)]
    if broken:
        print(
            "  WARNING: not comparable -- "
            + ", ".join(f"{k} {cfg_base.get(k)} -> {cfg_now.get(k)}" for k in broken)
        )
    changed = [k for k in TUNABLE if cfg_now.get(k) != cfg_base.get(k)]
    if changed:
        print(
            "  changed: " + ", ".join(f"{k} {cfg_base.get(k)} -> {cfg_now.get(k)}" for k in changed)
        )

    def delta(path: tuple[str, str], label: str) -> None:
        a = baseline["summary"].get(path[0], {}).get(path[1])
        b = current["summary"].get(path[0], {}).get(path[1])
        if a is None or b is None:
            return
        d = b - a
        arrow = "  " if abs(d) < 1e-9 else ("↑ " if d > 0 else "↓ ")
        print(f"  {label:18} {a:6.3f} -> {b:6.3f}  {arrow}{d:+.3f}")

    print()
    delta(("retrieval", "hit_rate"), "hit rate")
    delta(("retrieval", "mrr"), "MRR")
    delta(("retrieval", "page_recall"), "page recall")
    delta(("generation", "must_include_rate"), "must_include")
    delta(("judge", "correct_rate"), "judge correct")
    delta(("judge", "faithful_rate"), "judge faithful")

    base = {r["id"]: r for r in baseline["results"]}
    regressions, fixes = [], []
    for r in current["results"]:
        b = base.get(r["id"])
        if not b or r.get("error") or b.get("error"):
            continue
        for label, was, now in (
            ("retrieval", b["retrieval"]["hit"], r["retrieval"]["hit"]),
            (
                "must_include",
                (b.get("checks") or {}).get("must_include_ok"),
                (r.get("checks") or {}).get("must_include_ok"),
            ),
            (
                "judge",
                (b.get("judge") or {}).get("correctness") == "correct" if b.get("judge") else None,
                (r.get("judge") or {}).get("correctness") == "correct" if r.get("judge") else None,
            ),
        ):
            if was is True and now is False:
                regressions.append(f"{r['id']} ({label})")
            elif was is False and now is True:
                fixes.append(f"{r['id']} ({label})")
    if regressions:
        print(f"\n  regressed ({len(regressions)}): " + ", ".join(regressions))
    if fixes:
        print(f"  fixed     ({len(fixes)}): " + ", ".join(fixes))
    if not regressions and not fixes:
        print("\n  no per-entry changes")


# --- main ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--k", type=int, help="retrieval depth (default: RETRIEVAL_K)")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="score retrieval only; no LLM calls"
    )
    parser.add_argument("--judge", action="store_true", help="grade answers with an LLM judge")
    parser.add_argument("--judge-provider", help="judge provider (default: LLM_PROVIDER)")
    parser.add_argument(
        "--judge-model", help="judge model; a stronger one than the answering model is better"
    )
    parser.add_argument("--type", help="only entries of this type")
    parser.add_argument("--only", help="comma-separated entry ids")
    parser.add_argument("--limit", type=int, help="first N entries")
    parser.add_argument(
        "--delay", type=float, default=0.0, help="seconds between entries (rate limits)"
    )
    parser.add_argument(
        "--save-baseline", action="store_true", help="also store this run as the baseline"
    )
    parser.add_argument("--compare", action="store_true", help="diff against the stored baseline")
    args = parser.parse_args()

    if args.judge and args.retrieval_only:
        parser.error("--judge needs answers; drop --retrieval-only")

    settings = get_settings()
    entries = load_dataset()
    if args.type:
        entries = [e for e in entries if e["type"] == args.type]
    if args.only:
        wanted = set(args.only.split(","))
        entries = [e for e in entries if e["id"] in wanted]
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print("No entries match the filters.")
        return 1

    config = fingerprint(args, settings)
    judge_llm = build_judge(args, settings) if args.judge else None
    print(
        f"{len(entries)} entries | {config['mode']} | llm {config['llm_model']} | "
        f"k={config['k']} | {config['points']} points"
        + (f" | judge {config['judge_model']}" if args.judge else "")
    )
    if args.judge and config["judge_model"] == config["llm_model"]:
        print(
            "  note: the judge is the model being judged -- expect leniency; "
            "a stronger --judge-model gives a fairer grade"
        )
    print()

    results = []
    for i, entry in enumerate(entries, 1):
        r = run_entry(entry, args, judge_llm)
        results.append(r)
        print(progress_line(i, len(entries), r), flush=True)
        if args.delay and i < len(entries):
            time.sleep(args.delay)

    summary = summarize(results)
    print_summary(summary)

    run = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "summary": summary,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{config['mode']}.json"
    out.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(f"\nresults: {out.relative_to(BACKEND_ROOT)}")

    if args.compare:
        if BASELINE.exists():
            compare(run, json.loads(BASELINE.read_text()))
        else:
            print("\nNo baseline yet -- run with --save-baseline first.")
    if args.save_baseline:
        partial = len(entries) != len(load_dataset())
        if partial:
            print("\nNot saved as baseline: this run used a filtered subset of the dataset.")
        else:
            BASELINE.write_text(json.dumps(run, indent=2, ensure_ascii=False))
            print(f"baseline saved: {BASELINE.relative_to(BACKEND_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
