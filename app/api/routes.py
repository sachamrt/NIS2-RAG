"""HTTP routes. No business logic here -- everything delegates to app.rag."""

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.rag.chain import answer, answer_stream
from app.rag.documents import list_documents
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    DocumentsResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _history(request: ChatRequest) -> list[tuple[str, str]]:
    return [(m.role, m.content) for m in request.history]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    try:
        info = list_documents()
        reachable, points = True, info["points"]
    except Exception:  # noqa: BLE001 -- health must report, never raise
        reachable, points = False, 0

    model = getattr(settings, f"{settings.llm_provider}_llm_model", settings.llm_provider)
    return HealthResponse(
        status="ok" if reachable else "degraded",
        llm_provider=settings.llm_provider,
        llm_model=model,
        embedding_provider=settings.embedding_provider,
        collection=settings.qdrant_collection,
        points=points,
        qdrant_reachable=reachable,
    )


@router.get("/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    try:
        return DocumentsResponse(**list_documents())
    except Exception as exc:
        logger.exception("listing documents failed")
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Non-streaming answer -- handy for curl and tests."""
    try:
        text, sources = answer(request.question, _history(request), request.k)
    except Exception as exc:
        logger.exception("chat failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ChatResponse(answer=text, sources=[s.__dict__ for s in sources])


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-sent events: one `sources` event, then `token` events, then `done`.

    Sources are known before generation begins, so the UI can render citations
    while the answer is still being written.
    """

    def event(name: str, payload: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    def stream() -> Iterator[str]:
        try:
            sources, tokens = answer_stream(request.question, _history(request), request.k)
            yield event("sources", {"sources": [s.__dict__ for s in sources]})
            for token in tokens:
                yield event("token", {"token": token})
            yield event("done", {})
        except Exception as exc:  # the stream carries its own errors
            logger.exception("chat stream failed")
            yield event("error", {"detail": str(exc)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
