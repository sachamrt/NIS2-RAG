"""Request/response models for the chat API."""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list, max_length=20)
    k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    source: str
    page: int
    chunk_index: int
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class DocumentOut(BaseModel):
    source: str
    chunks: int


class DocumentsResponse(BaseModel):
    collection: str
    points: int
    documents: list[DocumentOut]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_model: str
    embedding_provider: str
    collection: str
    points: int
    qdrant_reachable: bool
