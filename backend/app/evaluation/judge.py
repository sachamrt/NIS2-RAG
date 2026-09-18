"""LLM-as-judge for answer quality.

Takes any LangChain chat model, so the judge goes through the same factory as
everything else and can be a different (ideally stronger) model than the one
being judged. A model grading its own answers is a known source of leniency.
"""

import json
import re
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

JUDGE_SYSTEM = """You grade answers produced by a retrieval-augmented assistant \
about the EU NIS2 directive. Be strict and literal. Reply with JSON only.

You are given the QUESTION, a REFERENCE answer written by an expert, the CONTEXT
the assistant was shown, and the assistant's ANSWER.

Grade three things independently:

1. "correctness" -- compare the ANSWER with the REFERENCE:
   - "correct":   conveys every key fact of the reference, contradicts none.
   - "partial":   right direction, but a key fact is missing or slightly wrong.
   - "incorrect": wrong, contradicts the reference, or misses the main point.
   Numbers, deadlines, dates and amounts must match exactly: "48 hours" is not
   "24 hours". Extra correct detail is fine.
   If the reference says the answer is NOT in the context, the answer is
   "correct" only if it says so and supplies no invented facts.

2. "faithful" -- is EVERY factual claim in the ANSWER supported by the CONTEXT?
   Judge against the CONTEXT only, not the reference and not your own
   knowledge. A claim that is true in the real world but absent from the
   CONTEXT makes the answer unfaithful. Figures, thresholds and legal
   references that appear nowhere in the CONTEXT are the usual failure.

3. "refused" -- does the ANSWER say the context does not contain the answer
   (fully or partly)?

Reply with exactly this JSON object and nothing else:
{{"correctness": "correct" | "partial" | "incorrect", "faithful": true | false, \
"refused": true | false, "reason": "<one sentence>"}}"""

JUDGE_HUMAN = """QUESTION:
{question}

REFERENCE:
{reference}

CONTEXT:
{context}

ANSWER:
{answer}"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])

_CORRECTNESS = {"correct", "partial", "incorrect"}


class JudgeError(ValueError):
    """The judge replied with something that is not the expected verdict."""


def parse_verdict(raw: str) -> dict:
    """Extract the verdict from the judge's reply.

    Small models often wrap JSON in prose or a ```json fence despite being told
    not to, so take the first {...} block rather than trusting the whole reply.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise JudgeError(f"no JSON object in judge reply: {raw[:200]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"invalid JSON from judge: {exc}") from exc

    correctness = str(data.get("correctness", "")).strip().lower()
    if correctness not in _CORRECTNESS:
        raise JudgeError(f"unexpected correctness {data.get('correctness')!r}")

    return {
        "correctness": correctness,
        "faithful": _as_bool(data.get("faithful")),
        "refused": _as_bool(data.get("refused")),
        "reason": str(data.get("reason", "")).strip(),
    }


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def judge(
    llm: "BaseChatModel", *, question: str, reference: str, context: str, answer: str
) -> dict:
    reply = (JUDGE_PROMPT | llm).invoke(
        {"question": question, "reference": reference, "context": context, "answer": answer}
    )
    return parse_verdict(reply.content)
