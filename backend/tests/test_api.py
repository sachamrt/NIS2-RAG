"""API contract tests. The LLM and Qdrant are stubbed so these run offline."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.chain import Source

FAKE_DOCS = {
    "collection": "nis2_test",
    "points": 7,
    "documents": [{"source": "nis2.pdf", "chunks": 5}, {"source": "annex.pdf", "chunks": 2}],
}
FAKE_SOURCES = [Source("nis2.pdf", 14, 0, 0.83, "Article 21 ...")]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.api.routes.list_documents", lambda: FAKE_DOCS)
    return TestClient(app)


def test_documents_lists_ingested_files(client):
    body = client.get("/api/documents").json()
    assert body["points"] == 7
    assert [d["source"] for d in body["documents"]] == ["nis2.pdf", "annex.pdf"]


def test_health_reports_providers_and_collection(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["qdrant_reachable"] is True
    assert body["llm_provider"] and body["llm_model"]


def test_health_degrades_instead_of_failing(monkeypatch):
    def boom():
        raise ConnectionError("qdrant down")

    monkeypatch.setattr("app.api.routes.list_documents", boom)
    body = TestClient(app).get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["qdrant_reachable"] is False


def test_documents_returns_503_when_qdrant_is_down(monkeypatch):
    def boom():
        raise ConnectionError("qdrant down")

    monkeypatch.setattr("app.api.routes.list_documents", boom)
    assert TestClient(app).get("/api/documents").status_code == 503


def test_chat_returns_answer_and_sources(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.answer", lambda q, h, k: ("Entities must ...", FAKE_SOURCES)
    )
    body = client.post("/api/chat", json={"question": "what measures?"}).json()
    assert body["answer"] == "Entities must ..."
    assert body["sources"][0]["source"] == "nis2.pdf"
    assert body["sources"][0]["page"] == 14


def test_chat_passes_history_through(client, monkeypatch):
    seen = {}

    def spy(question, history, k):
        seen["history"] = history
        return "ok", []

    monkeypatch.setattr("app.api.routes.answer", spy)
    client.post(
        "/api/chat",
        json={
            "question": "and for important entities?",
            "history": [
                {"role": "user", "content": "what measures?"},
                {"role": "assistant", "content": "Entities must ..."},
            ],
        },
    )
    assert seen["history"] == [
        ("user", "what measures?"),
        ("assistant", "Entities must ..."),
    ]


def test_chat_rejects_empty_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_stream_emits_sources_then_tokens_then_done(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.answer_stream",
        lambda q, h, k: (FAKE_SOURCES, iter(["Entities ", "must ", "act."])),
    )
    with client.stream("POST", "/api/chat/stream", json={"question": "?"}) as res:
        body = "".join(res.iter_text())

    events = [f for f in body.split("\n\n") if f.strip()]
    names = [f.split("\n")[0].removeprefix("event: ") for f in events]
    assert names == ["sources", "token", "token", "token", "done"]

    first = json.loads(events[0].split("\n")[1].removeprefix("data: "))
    assert first["sources"][0]["page"] == 14
    tokens = [json.loads(e.split("\n")[1].removeprefix("data: "))["token"] for e in events[1:4]]
    assert "".join(tokens) == "Entities must act."


def test_stream_reports_errors_as_an_event_not_a_500(client, monkeypatch):
    def boom(q, h, k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr("app.api.routes.answer_stream", boom)
    with client.stream("POST", "/api/chat/stream", json={"question": "?"}) as res:
        assert res.status_code == 200  # headers already sent; error rides the stream
        body = "".join(res.iter_text())
    assert "event: error" in body and "rate limited" in body
