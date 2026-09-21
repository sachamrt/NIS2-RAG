# NIS2-RAG

A retrieval-augmented generation (RAG) assistant for the **NIS2 directive**
(Directive (EU) 2022/2555 on cybersecurity). Ask a question in plain language
and get an answer grounded in the directive, with citations down to the PDF page
(`[CELEX_32022L2555_EN_TXT.pdf, p.63]`).

It is a learning project for AI engineering, built in three phases:

1. **Basic RAG** on the NIS2 PDF corpus (extensible to more PDFs). *Done.*
2. **Production-grade RAG**: hybrid search, structure-aware chunking,
   reranking, generation guards, observability. *In progress.*
3. **LangGraph agent** with web search on top. *Planned.*

## Features

- **Streaming chat**: answers stream over Server-Sent Events. Sources are sent
  before the first token, so citations appear while the answer is still being
  written.
- **Page-level citations**: every answer names the file and page it relies on.
- **Hybrid search**: dense embeddings combined with BM25 sparse vectors in
  Qdrant, fused with Reciprocal Rank Fusion.
- **Pluggable providers**: LLM, embeddings and sparse model are selected from
  `.env`. Mistral, OpenAI, Anthropic or local Ollama, with no code changes.
- **Document management**: upload and delete PDFs from the UI. The curated
  corpus in `data/raw_pdfs/` is protected from deletion.

## Stack

| Layer | Technology |
|---|---|
| Backend API | Python 3.12, FastAPI, Uvicorn |
| RAG framework | LangChain (`langchain-core`, `langchain-qdrant`) |
| Vector database | Qdrant (Docker), dense + sparse named vectors |
| LLM | Mistral by default (`ministral-8b-latest`); OpenAI, Anthropic, Ollama supported |
| Embeddings | `mistral-embed` by default; OpenAI or Ollama (`nomic-embed-text`) supported |
| Sparse / BM25 | FastEmbed `Qdrant/bm25` (runs client-side; Qdrant applies IDF) |
| PDF parsing | `pypdf` |
| Configuration | `pydantic-settings`, values read from `.env` |
| Frontend | React 19 + Vite, plain CSS (no UI framework) |
| Tooling | pytest, ruff (Python), oxlint (JS) |

## Architecture

```
            ┌──────────────┐   /api (Vite proxy)   ┌─────────────────────┐
  Browser → │ React + Vite │ ────────────────────→ │ FastAPI (backend/)  │
            └──────────────┘        SSE ←──────────│  api → rag → core   │
                                                   └──────────┬──────────┘
                                                              │
                          ┌───────────────────────────────────┼──────────────┐
                          ▼                                   ▼              ▼
                   Qdrant (Docker)                   LLM provider     Embedding provider
             dense + BM25 sparse vectors           (via llm_factory)  (via embeddings_factory)
```

**Ingestion**: PDF → `pypdf` page text → chunking → dense + sparse embeddings →
upsert into Qdrant. Chunk IDs are deterministic (`uuid5` of source, page, chunk
index and content hash), so re-ingesting an unchanged PDF does not create
duplicates.

**Query**: question → hybrid retrieval (top *k*, default 4) → context formatted
with file and page labels → prompt → LLM → streamed answer and sources.

### Repository layout

```
backend/
  app/
    api/          FastAPI routes only, no business logic
    core/         settings, provider factories (LLM, embeddings, sparse), Qdrant client, paths
    ingestion/    PDF load → split → embed → upsert, plus upload/delete
    rag/          retriever, chain, prompts, document listing
    evaluation/   eval metrics and the LLM judge
    schemas/      Pydantic request/response models
  eval/           evaluation dataset (nis2_eval.jsonl), baseline, docs
  scripts/        CLI entrypoints: ingestion, provider checks, eval runner
  tests/
frontend/
  src/            React app (Chat, Sidebar, Sources components; api.js SSE client)
data/raw_pdfs/    curated PDF corpus, added by hand
docker-compose.yml  Qdrant
```

Downstream code only depends on LangChain's `BaseChatModel` / `Embeddings`
interfaces. Providers are always built through `app/core/*_factory.py`, and
vendor SDKs are imported lazily, so you only need to install the one you use.

## Getting started

### Prerequisites

- Python 3.12+
- Node.js (for the frontend)
- Docker (for Qdrant)
- An API key for your chosen provider (Mistral by default), or a local
  [Ollama](https://ollama.com) install

### 1. Configure

```bash
cp .env.example .env
```

Set at least `MISTRAL_API_KEY` (or change `LLM_PROVIDER` / `EMBEDDING_PROVIDER`
and give that provider's key). To turn on hybrid search, add:

```bash
SPARSE_PROVIDER=bm25                   # none = dense-only
QDRANT_COLLECTION=nis2_mistral_hybrid  # a hybrid collection needs a sparse slot
```

> A Qdrant collection belongs to one embedder (dimensions differ per provider)
> and one sparse setting. Changing `EMBEDDING_PROVIDER`, `SPARSE_PROVIDER` or the
> chunking means using a new `QDRANT_COLLECTION` and re-ingesting.

### 2. Start Qdrant

```bash
docker compose up -d
```

### 3. Install the backend and ingest the corpus

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

python backend/scripts/check_providers.py   # verify LLM + embeddings are reachable
python backend/scripts/ingest_cli.py        # index data/raw_pdfs/
python backend/scripts/ingest_cli.py --stats
```

### 4. Run the API and the UI

```bash
cd backend && uvicorn app.main:app --reload    # http://localhost:8000
cd frontend && npm install && npm run dev      # http://localhost:5173
```

The Vite dev server proxies `/api` to `localhost:8000`, so there is no CORS setup
in development.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Provider, model, collection, point count, Qdrant reachability |
| `GET` | `/api/documents` | Indexed documents, with a `deletable` flag per row |
| `POST` | `/api/documents` | Upload and index a PDF (multipart `file`) |
| `DELETE` | `/api/documents/{filename}` | Remove an uploaded document's chunks (403 for curated files) |
| `POST` | `/api/chat` | Non-streaming answer with sources |
| `POST` | `/api/chat/stream` | SSE stream: `sources`, then `token` events, then `done` (or `error`) |

Chat request body: `{ "question": "...", "history": [{ "role": "...", "content": "..." }], "k": 4 }`.

Uploaded PDFs are indexed and then discarded; only their chunks are kept in
Qdrant. Rebuilding the index restores `data/raw_pdfs/` only, so uploads have to
be uploaded again.

## Evaluation

The evaluation set (`backend/eval/nis2_eval.jsonl`) holds 46 questions written by
reading the directive: lookups, definitions, multi-part lists, comparisons,
partial and unanswerable questions, and a few deliberate hallucination traps.
Every ground-truth page carries a verbatim quote, and the tests check each
quote against the real PDF.

```bash
# Retrieval only (embeddings only, no LLM cost)
python backend/scripts/run_eval.py --retrieval-only

# Full run with an LLM judge, compared to the committed baseline
python backend/scripts/run_eval.py --judge --judge-model ministral-14b-latest --compare
```

## Development

```bash
cd backend && pytest          # tests
cd backend && ruff check .    # Python lint
cd frontend && npm run lint   # JS lint (oxlint)

python backend/scripts/check_llm.py --provider mistral --model ministral-8b-latest
python backend/scripts/ingest_cli.py --clear            # drop the whole index
python backend/scripts/ingest_cli.py --clear file.pdf   # drop one document
```
