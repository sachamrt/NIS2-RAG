# NIS2 evaluation set

`nis2_eval.jsonl` — 46 questions about the NIS2 directive with known-correct
answers, used to measure the RAG pipeline. One JSON object per line.

Ground truth was written by **reading the directive**, never by recording what
retrieval returns — otherwise today's retrieval errors would become the answer
key and the score could never expose them.

## Fields

| Field | Meaning |
|---|---|
| `id` | stable, unique slug |
| `type` | see below |
| `question` | what gets sent to the pipeline |
| `reference_answer` | the correct answer, with Article references |
| `evidence` | `[{source, page, quote}]` — the primary page(s) holding the answer, plus a verbatim quote proving it. On a multi-page question **all** are needed. |
| `also_answered_by` | `[{source, page, quote}]` — pages that **alone** hold the complete answer, usually a recital restating an Article. Retrieving one counts as a full hit. |
| `must_include` | groups of acceptable phrasings; a correct answer contains ≥1 from **every** group |
| `must_not_include` | strings whose presence signals a hallucination |

Pages are **PDF page numbers** (1-based, as the loader cites them), not the
Official Journal's printed page numbers.

## Types

| Type | n | Tests |
|---|---|---|
| `lookup` | 25 | a single fact: a deadline, amount, rule |
| `definition` | 5 | an Article 6 / 23(3) definition |
| `multi_part` | 5 | a list that must be reproduced in full |
| `multi_page` | 3 | an answer split across two pages |
| `comparison` | 2 | two similar things the retriever may confuse |
| `recital` | 1 | content that lives only in the preamble (pp. 1–28) |
| `partial` | 2 | the topic is discussed, but the specific question is not answered |
| `unanswerable` | 3 | not in the document; the model must not invent an answer |

## Deliberate traps

- **`scope-size-threshold`** — NIS2 defers to Recommendation 2003/361/EC and
  states no euro or headcount figures. An answer quoting "€50 million" or
  "250 employees" is reciting from memory: `must_not_include` catches it. An
  early prompt produced exactly this error ("€40 million").
- **`waste-water-vs-waste-management`** — near-identical wording, different
  annexes (I vs II).
- **`deadline-trust-service-provider`** and **`deadline-csirt-response`** — both
  are "24 hours", but for different parties; easy to conflate with the early warning.
- **`unanswerable-gdpr-fine`** — Art. 35 discusses GDPR fines without stating
  their amount, so the model is tempted to fill it in.
- **`partial-ransom-payment`** — ransomware is discussed (recital 54), payment
  legality is not. A correct answer says both.

## Alternative pages, and how they were found

The recitals (pp. 1–28) often restate an Article in plain language, so a
question can be fully answered on a page other than the Article. The first
retrieval run scored three such cases as misses: the retriever had found a
recital holding the whole answer, and the dataset only knew the Article.

To fix that without biasing the set toward the current retriever, every
question was screened against **every** page (not just the pages retrieval
happened to return), and each candidate was **read**. 12 were kept. More were
rejected because they share keywords but state a different rule — e.g. p.20's
"24 hours" is the entity's early-warning deadline, not the CSIRT's response
deadline, and p.54's "72 hours" is the domain-data deadline, not incident
notification. A page is only an alternative if it holds the complete answer.

## Running

```bash
python scripts/run_eval.py --retrieval-only        # free: embeddings only
python scripts/run_eval.py --judge --judge-model ministral-14b-latest
python scripts/run_eval.py --judge --judge-model ministral-14b-latest --compare
```

Each run writes `results/<timestamp>-<mode>.json` (gitignored). `baseline.json`
is the reference run and is committed. `--save-baseline` refuses a filtered run.

Every run records a config fingerprint. `--compare` flags a change in
`dataset_sha` or `judge_model` as **not comparable** — that measures a change of
ruler, not of the system. Changes to `llm_model`, `k`, chunking or `points` are
reported as what you tuned.

## Scoring

- **Retrieval** — hit if any retrieved chunk's `(source, page)` is in
  `evidence` or `also_answered_by`. Match on page, **never on chunk ids** —
  those change with `CHUNK_SIZE`, and this set must survive that tuning.
- **Generation** — `must_include` / `must_not_include` are a cheap first pass.
  They are substring heuristics and will misjudge some phrasings; an LLM judge
  comparing against `reference_answer` is authoritative.
- **Unanswerable** — score whether anything was *invented*, via the judge's
  correctness grade. Not whether the model "refused": for "Does NIS2 require
  cyber insurance?" the right answer is simply "No", which is a real answer,
  not a refusal. The first baseline's 25% "refusal accuracy" was an artefact of
  measuring the wrong thing.

## Trusting the numbers

**Neither the judge nor the phrase checks is reliable alone.** In the first
judged run, three entries had the two signals contradicting each other, and
each side was wrong at least once:

| Entry | Wrong | What happened |
|---|---|---|
| `waste-water-vs-waste-management` | judge | Answer was right; the judge claimed waste water is in Annex II, contradicting the reference it was handed. The near-identical terms that trap the retriever trapped the judge. |
| `management-body-duties` | judge | Judge said "fully matches"; the answer omitted that management bodies can be held liable. Lenient. |
| `voluntary-notification` | phrases | Answer said **"No"** — the opposite of correct — but contained "does not mention any *voluntar*y reporting", so the substring check passed. Substrings cannot see negation. |

| `digital-providers` | judge | Answer said Annex II does **not** cover digital providers — flatly wrong. Retrieval returned only the start of Annex II; the judge reasoned from that partial context instead of the reference and graded it correct. |
| `scope-size-threshold` | judge | Answer stated "€50 million" / "€43 million", which NIS2 never states. The judge marked it **faithful** — and in its reasoning corrected the figures' formatting from its own knowledge. |

The last row is structural, not a bad day: **a judge that knows a fact is true
cannot reliably notice it is missing from the context.** That is precisely the
recited-from-memory failure the faithfulness score exists to catch. For known
traps, the deterministic `must_not_include` markers are the ground truth, and a
marker plus a "faithful" verdict is itself reported as a contradiction.

The runner prints these contradictions after every judged run. Read them
before trusting a score, and before believing a baseline moved.

**Absence checks must be broad.** `unanswerable-iso27001` was mislabelled: an
exact-string search for "ISO 27001" found nothing, but recital text on p.16
refers to the "ISO/IEC 27000 series". The model found and cited it. It is now
`partial-iso27001`. When labelling something unanswerable, search for
variants, not the literal phrase.

## Editing

`backend/tests/test_eval_dataset.py` checks every quote is really on its stated
page of the PDF, and that no hallucination marker actually occurs in the
directive. Run `pytest tests/test_eval_dataset.py` after any change.
