"""Retrieval-augmented answering.

Deliberately thin: retrieve -> format context -> prompt -> LLM. The LLM and the
embedder both arrive through the factories, so the chain never names a provider.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.core.llm_factory import get_llm
from app.core.vectorstore import get_vectorstore
from app.rag.prompts import ANSWER_PROMPT, format_context

if TYPE_CHECKING:
    from langchain_core.documents import Document


@dataclass
class Source:
    source: str
    page: int
    chunk_index: int
    score: float
    excerpt: str


def _to_source(doc: "Document", score: float) -> Source:
    return Source(
        source=doc.metadata.get("source", "?"),
        page=int(doc.metadata.get("page", 0)),
        chunk_index=int(doc.metadata.get("chunk_index", 0)),
        score=round(float(score), 4),
        excerpt=doc.page_content[:300],
    )


def retrieve(question: str, k: int | None = None) -> list[tuple["Document", float]]:
    k = k or get_settings().retrieval_k
    return get_vectorstore().similarity_search_with_score(question, k=k)


def _prompt_inputs(question: str, history: list[tuple[str, str]] | None, k: int | None):
    hits = retrieve(question, k)
    documents = [doc for doc, _ in hits]
    messages = [(role, content) for role, content in (history or [])]
    return hits, {
        "context": format_context(documents),
        "question": question,
        "history": messages,
    }


def answer(
    question: str, history: list[tuple[str, str]] | None = None, k: int | None = None
) -> tuple[str, list[Source]]:
    """Answer a question; returns (answer text, sources)."""
    hits, inputs = _prompt_inputs(question, history, k)
    chain = ANSWER_PROMPT | get_llm()
    reply = chain.invoke(inputs)
    return reply.content, [_to_source(doc, score) for doc, score in hits]


def answer_stream(
    question: str, history: list[tuple[str, str]] | None = None, k: int | None = None
) -> tuple[list[Source], Iterator[str]]:
    """Sources are known before generation starts, so return them immediately
    and hand back an iterator of answer tokens."""
    hits, inputs = _prompt_inputs(question, history, k)
    sources = [_to_source(doc, score) for doc, score in hits]

    def tokens() -> Iterator[str]:
        for piece in (ANSWER_PROMPT | get_llm()).stream(inputs):
            if piece.content:
                yield piece.content

    return sources, tokens()
