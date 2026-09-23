"""Scoring for the evaluation set. Pure functions: no network, no LLM.

Retrieval is scored on (source, page) and never on chunk ids, because chunk ids
change with CHUNK_SIZE -- and the eval set has to survive exactly that tuning.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from app.core.paths import BACKEND_ROOT

EVAL_FILE = BACKEND_ROOT / "eval" / "nis2_eval.jsonl"

# Phrasings a model uses when the context does not hold the answer. A heuristic:
# the LLM judge is authoritative when it runs.
_REFUSAL = re.compile(
    r"not (?:in|contained in|found in|present in|included in) "
    r"the (?:context|document|directive|text)"
    r"|(?:does not|doesn't|do not) (?:explicitly )?"
    r"(?:contain|mention|specify|state|address|say|provide|include|cover|refer"
    r"|require|mandate|impose|set)"
    r"|no (?:information|mention|reference|details?)"
    r"|not (?:specified|mentioned|stated|addressed|covered|provided)"
    r"|(?:cannot|can't|unable to) (?:find|answer|determine)",
    re.IGNORECASE,
)


def load_dataset(path: Path = EVAL_FILE) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def is_answerable(entry: dict) -> bool:
    """Entries with evidence can be scored on retrieval."""
    return bool(entry["evidence"])


# --- retrieval -----------------------------------------------------------


def retrieval_metrics(entry: dict, retrieved: list[tuple[str, int | list[int]]]) -> dict:
    """Score retrieved (source, page) pairs, in rank order, against the evidence.

    A hit may carry a list of pages: a structured chunk (one Article paragraph)
    can straddle a page break, and it holds text from every page it lists.

    Two kinds of expected page:
    - ``evidence``: the answer's primary pages. On a multi-page question they
      are *all* needed -- each holds part of the answer.
    - ``also_answered_by``: pages that *alone* hold the complete answer, usually
      a recital restating an Article. Retrieving one means the full answer
      reached the model, even if no primary page did.

    - hit:          any expected page (either kind) retrieved
    - rank:         1-based rank of the first expected page, else None
    - rr:           reciprocal rank (1/rank), 0 on a miss -- averaged, this is MRR
    - page_recall:  1.0 if an alternative page was retrieved; otherwise the share
                    of distinct primary pages retrieved. Below 1.0 on a
                    multi-page question means part of the answer never reached
                    the model.
    """
    primary = {(ev["source"], ev["page"]) for ev in entry["evidence"]}
    alternatives = {(ev["source"], ev["page"]) for ev in entry.get("also_answered_by", [])}
    if not primary:
        return {"hit": None, "rank": None, "rr": None, "page_recall": None}

    hits = [
        {(source, page) for page in (pages if isinstance(pages, list) else [pages])}
        for source, pages in retrieved
    ]
    expected = primary | alternatives
    rank = next((i for i, pairs in enumerate(hits, 1) if pairs & expected), None)
    found = set().union(*hits)
    recall = 1.0 if alternatives & found else len(primary & found) / len(primary)
    return {
        "hit": rank is not None,
        "rank": rank,
        "rr": 1.0 / rank if rank else 0.0,
        "page_recall": recall,
    }


# --- generation, heuristic -----------------------------------------------


def phrase_checks(entry: dict, answer: str) -> dict:
    """Cheap first-pass answer check against must_include / must_not_include.

    Each must_include item is a group of acceptable phrasings: the answer passes
    if it contains at least one phrasing from *every* group. Case-insensitive
    substring match -- it will misjudge some phrasings; the judge is authoritative.
    """
    text = answer.lower()
    missing = [g for g in entry["must_include"] if not any(alt.lower() in text for alt in g)]
    violations = [term for term in entry["must_not_include"] if term.lower() in text]
    return {
        "must_include_ok": not missing if entry["must_include"] else None,
        "missing": missing,
        "hallucination_flags": violations,
        "refused": looks_like_refusal(answer),
    }


def looks_like_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer))


# --- aggregation ---------------------------------------------------------


def _rate(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(bool(v) for v in values) / len(values), 3) if values else None


def _avg(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(mean(values), 3) if values else None


def summarize(results: list[dict]) -> dict:
    """Roll per-entry results up into the headline numbers."""
    ok = [r for r in results if not r.get("error")]
    retr = [r for r in ok if r["retrieval"]["hit"] is not None]
    gen = [r for r in ok if r.get("checks")]
    unans = [r for r in gen if r["type"] == "unanswerable"]
    judged = [r for r in ok if r.get("judge")]

    summary = {
        "entries": len(results),
        "errors": len(results) - len(ok),
        "retrieval": {
            "n": len(retr),
            "hit_rate": _rate([r["retrieval"]["hit"] for r in retr]),
            "mrr": _avg([r["retrieval"]["rr"] for r in retr]),
            "page_recall": _avg([r["retrieval"]["page_recall"] for r in retr]),
        },
        "generation": {
            "n": len(gen),
            "must_include_rate": _rate([r["checks"]["must_include_ok"] for r in gen]),
            "hallucination_flags": sum(len(r["checks"]["hallucination_flags"]) for r in gen),
            "refusal_rate_unanswerable": _rate([r["checks"]["refused"] for r in unans]),
        },
    }
    if judged:
        answerable = [r for r in judged if r["type"] != "unanswerable"]
        refusals = [r for r in judged if r["type"] == "unanswerable"]
        summary["judge"] = {
            "n": len(judged),
            "correct_rate": _rate([r["judge"]["correctness"] == "correct" for r in answerable]),
            "partial_rate": _rate([r["judge"]["correctness"] == "partial" for r in answerable]),
            "faithful_rate": _rate([r["judge"]["faithful"] for r in judged]),
            "unanswerable_correct": _rate(
                [r["judge"]["correctness"] == "correct" for r in refusals]
            ),
        }

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_type[r["type"]].append(r)
    summary["by_type"] = {
        t: {
            "n": len(rs),
            "hit_rate": _rate([r["retrieval"]["hit"] for r in rs]),
            "must_include_rate": _rate(
                [(r.get("checks") or {}).get("must_include_ok") for r in rs]
            ),
            **(
                {"correct_rate": _rate([r["judge"]["correctness"] == "correct" for r in rs])}
                if all(r.get("judge") for r in rs)
                else {}
            ),
        }
        for t, rs in sorted(by_type.items())
    }
    summary["disagreements"] = disagreements(ok)
    return summary


def disagreements(results: list[dict]) -> list[dict]:
    """Entries where the deterministic checks and the judge contradict each other.

    Neither signal is reliable alone. The judge misreads near-identical terms,
    can be lenient, and -- structurally -- cannot reliably spot a claim that is
    true in the world but absent from the context, because it knows the claim
    too. Substring checks miss paraphrases and cannot see negation. Where they
    disagree, one of them is wrong: these are the entries worth a human read.
    """
    flagged = []
    for r in results:
        judge, checks = r.get("judge"), r.get("checks") or {}
        if not judge:
            continue
        inc = checks.get("must_include_ok")
        if inc and judge["correctness"] == "incorrect":
            flagged.append({"id": r["id"], "phrases": "pass", "judge": "incorrect"})
        elif inc is False and judge["correctness"] == "correct":
            flagged.append(
                {
                    "id": r["id"],
                    "phrases": "fail",
                    "judge": "correct",
                    "missing": checks.get("missing"),
                }
            )
        # A known hallucination marker is deterministic evidence of a claim from
        # outside the context. A judge calling that answer faithful is wrong.
        if checks.get("hallucination_flags") and judge["faithful"]:
            flagged.append(
                {
                    "id": r["id"],
                    "phrases": "halluc",
                    "judge": "faithful",
                    "missing": checks["hallucination_flags"],
                }
            )
    return flagged
