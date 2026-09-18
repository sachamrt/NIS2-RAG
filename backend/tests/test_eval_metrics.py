"""Scoring and judge parsing. Pure functions: no network, no LLM."""

import pytest

from app.evaluation.judge import JudgeError, parse_verdict
from app.evaluation.metrics import (
    looks_like_refusal,
    phrase_checks,
    retrieval_metrics,
    summarize,
)

SRC = "nis2.pdf"


def entry(pages=(50,), must=(), must_not=(), type_="lookup"):
    return {
        "id": "x",
        "type": type_,
        "evidence": [{"source": SRC, "page": p, "quote": "q"} for p in pages],
        "must_include": [list(g) for g in must],
        "must_not_include": list(must_not),
    }


# --- retrieval -----------------------------------------------------------


def test_hit_at_first_rank():
    m = retrieval_metrics(entry(), [(SRC, 50), (SRC, 12)])
    assert m == {"hit": True, "rank": 1, "rr": 1.0, "page_recall": 1.0}


def test_hit_at_lower_rank_lowers_reciprocal_rank():
    m = retrieval_metrics(entry(), [(SRC, 3), (SRC, 7), (SRC, 50)])
    assert m["rank"] == 3 and m["rr"] == pytest.approx(1 / 3)


def test_miss():
    m = retrieval_metrics(entry(), [(SRC, 1), (SRC, 2)])
    assert m == {"hit": False, "rank": None, "rr": 0.0, "page_recall": 0.0}


def test_right_page_wrong_document_is_a_miss():
    assert retrieval_metrics(entry(), [("other.pdf", 50)])["hit"] is False


def test_page_recall_on_multi_page_answer():
    # Half the answer reached the model: a hit, but recall exposes the gap.
    m = retrieval_metrics(entry(pages=(47, 48)), [(SRC, 48), (SRC, 48), (SRC, 9)])
    assert m["hit"] is True
    assert m["page_recall"] == 0.5


def _with_alt(e, page):
    return {**e, "also_answered_by": [{"source": SRC, "page": page, "quote": "q"}]}


def test_alternative_page_counts_as_a_hit():
    m = retrieval_metrics(_with_alt(entry(), 20), [(SRC, 3), (SRC, 20)])
    assert m["hit"] is True and m["rank"] == 2


def test_alternative_page_gives_full_recall_on_multi_page_question():
    # A recital restating the whole answer covers what both Articles hold.
    e = _with_alt(entry(pages=(56, 58)), 25)
    assert retrieval_metrics(e, [(SRC, 25)])["page_recall"] == 1.0


def test_without_alternative_recall_is_still_partial():
    e = _with_alt(entry(pages=(56, 58)), 25)
    assert retrieval_metrics(e, [(SRC, 56)])["page_recall"] == 0.5


def test_unanswerable_has_no_retrieval_score():
    m = retrieval_metrics(entry(pages=()), [(SRC, 1)])
    assert m == {"hit": None, "rank": None, "rr": None, "page_recall": None}


# --- phrase checks -------------------------------------------------------


def test_every_group_must_match_one_alternative():
    e = entry(must=[("24 hours", "24h"), ("72 hours", "72h")])
    assert phrase_checks(e, "Early warning in 24h, notification within 72 HOURS.")[
        "must_include_ok"
    ]
    result = phrase_checks(e, "Early warning within 24 hours.")
    assert result["must_include_ok"] is False
    assert result["missing"] == [["72 hours", "72h"]]


def test_hallucination_flags():
    e = entry(must_not=["250 employees", "€50"])
    flags = phrase_checks(e, "Medium-sized means fewer than 250 employees.")["hallucination_flags"]
    assert flags == ["250 employees"]


def test_no_must_include_is_not_scored():
    assert phrase_checks(entry(), "anything")["must_include_ok"] is None


@pytest.mark.parametrize(
    "text",
    [
        "The context does not mention cyber insurance.",
        "This information is not in the context.",
        "The directive doesn't specify the GDPR fine amount.",
        "There is no information about ISO 27001 in the provided document.",
        "I cannot find a national law in the context.",
        "The NIS2 directive does not explicitly require entities to hold cyber insurance.",
    ],
)
def test_refusals_are_recognised(text):
    assert looks_like_refusal(text)


def test_a_plain_answer_is_not_a_refusal():
    assert not looks_like_refusal(
        "Entities must send an early warning within 24 hours [nis2.pdf, p.50]."
    )


# --- judge parsing -------------------------------------------------------


def test_parses_clean_json():
    v = parse_verdict(
        '{"correctness": "correct", "faithful": true, "refused": false, "reason": "ok"}'
    )
    assert v == {"correctness": "correct", "faithful": True, "refused": False, "reason": "ok"}


def test_parses_json_wrapped_in_a_fence_and_prose():
    raw = (
        "Here is my verdict:\n```json\n"
        '{"correctness": "Partial", "faithful": "false", "refused": "no"}\n```'
    )
    v = parse_verdict(raw)
    assert v["correctness"] == "partial"  # normalised case
    assert v["faithful"] is False  # string booleans coerced
    assert v["refused"] is False


@pytest.mark.parametrize(
    "raw",
    ["no json here", '{"correctness": "great", "faithful": true}', "{not json}"],
)
def test_rejects_bad_verdicts(raw):
    with pytest.raises(JudgeError):
        parse_verdict(raw)


# --- aggregation ---------------------------------------------------------


def _result(id_, type_, hit, rr, inc, refused=False, flags=(), judge=None, error=None):
    r = {
        "id": id_,
        "type": type_,
        "retrieval": {"hit": hit, "rank": None, "rr": rr, "page_recall": 1.0 if hit else 0.0},
        "checks": {
            "must_include_ok": inc,
            "missing": [],
            "hallucination_flags": list(flags),
            "refused": refused,
        },
    }
    if judge:
        r["judge"] = judge
    if error:
        r["error"] = error
    return r


def test_summary_excludes_unanswerable_from_retrieval_and_errors_from_everything():
    results = [
        _result("a", "lookup", True, 1.0, True),
        _result("b", "lookup", False, 0.0, False),
        _result("c", "unanswerable", None, None, None, refused=True),
        _result("d", "lookup", True, 1.0, True, error="429"),
    ]
    s = summarize(results)
    assert s["errors"] == 1
    assert s["retrieval"]["n"] == 2  # c is unanswerable, d errored
    assert s["retrieval"]["hit_rate"] == 0.5
    assert s["retrieval"]["mrr"] == 0.5
    assert s["generation"]["refusal_rate_unanswerable"] == 1.0


def test_judge_rates_split_answerable_from_refusals():
    v = {"faithful": True, "refused": False, "reason": ""}
    results = [
        _result("a", "lookup", True, 1.0, True, judge={**v, "correctness": "correct"}),
        _result("b", "lookup", True, 1.0, True, judge={**v, "correctness": "partial"}),
        _result(
            "c",
            "unanswerable",
            None,
            None,
            None,
            judge={**v, "correctness": "correct", "refused": True},
        ),
    ]
    j = summarize(results)["judge"]
    assert j["correct_rate"] == 0.5  # over answerable only
    assert j["partial_rate"] == 0.5
    assert j["unanswerable_correct"] == 1.0


def test_disagreements_flag_contradictions_only():
    v = {"faithful": True, "refused": False, "reason": ""}
    results = [
        # phrase pass, judge incorrect -> flag (the judge misread waste water/management)
        _result("a", "comparison", True, 1.0, True, judge={**v, "correctness": "incorrect"}),
        # phrase fail, judge correct -> flag (the judge missed an omitted fact)
        _result("b", "multi_page", True, 1.0, False, judge={**v, "correctness": "correct"}),
        # agreement either way -> not flagged
        _result("c", "lookup", True, 1.0, True, judge={**v, "correctness": "correct"}),
        _result("d", "lookup", True, 1.0, False, judge={**v, "correctness": "incorrect"}),
        # partial is not a contradiction of a phrase pass
        _result("e", "lookup", True, 1.0, True, judge={**v, "correctness": "partial"}),
    ]
    flagged = summarize(results)["disagreements"]
    assert [d["id"] for d in flagged] == ["a", "b"]


def test_hallucination_marker_contradicts_a_faithful_verdict():
    # The judge knows the EUR 50M threshold is real, so it may miss that the
    # context never states it. The deterministic marker is the ground truth.
    judge = {"correctness": "partial", "faithful": True, "refused": False, "reason": ""}
    r = _result("scope", "lookup", True, 1.0, True, flags=["50 million"], judge=judge)
    flagged = summarize([r])["disagreements"]
    assert flagged == [
        {"id": "scope", "phrases": "halluc", "judge": "faithful", "missing": ["50 million"]}
    ]
