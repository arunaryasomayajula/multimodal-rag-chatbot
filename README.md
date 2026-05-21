# RAG Chatbot — Multi-Modal Retrieval-Augmented Generation

A fully open source, locally-hosted chatbot that answers questions by retrieving relevant content from your documents using Retrieval-Augmented Generation (RAG). Built with Ollama, Qdrant, FastAPI, and Open WebUI — no external API keys required.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Supported File Types](#supported-file-types)
- [Development](#development)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)

---

## Overview

This chatbot ingests your documents, indexes them in a vector store, and uses a local LLM to answer questions grounded in that content. Every answer includes citations back to the source document and page number.

**Key features:**
- Fully local — runs on your machine, no data leaves your network
- PDF ingestion with OCR fallback for scanned documents
- Hybrid retrieval: dense vector search + BM25 keyword search fused with Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking for higher answer quality
- Multi-turn conversation memory per session
- Async file ingestion via Celery — upload and continue while processing happens in the background
- Source citations in every response (`[source: filename, p.N]`)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                       │
│                                                               │
│  .txt / .md ──→ TextLoader                                   │
│  .pdf ───────→ pymupdf (text) + pdfplumber (tables)          │
│  (scanned)  ──→ Tesseract OCR fallback                       │
│                      ↓                                        │
│              RecursiveCharacterTextSplitter                   │
│              (512 tokens, 64 token overlap)                   │
│                      ↓                                        │
│              nomic-embed-text via Ollama                      │
│                      ↓                                        │
│     Qdrant ←── vectors + payload    PostgreSQL ←── metadata  │
└──────────────────────────────────────────────────────────────┘
                            ↕
┌──────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                          │
│                                                               │
│  User query                                                   │
│    → embed (nomic-embed-text)                                 │
│    → Qdrant dense search (top 20)                            │
│    + BM25 keyword search        (top 20)                     │
│    → RRF fusion                 (top 20 combined)            │
│    → Cross-encoder rerank       (top 5)                      │
│    → llama3.1:8b via Ollama                                  │
│    → Response + citations                                     │
└──────────────────────────────────────────────────────────────┘
                            ↕
┌──────────────────────────────────────────────────────────────┐
│   FastAPI (port 8000)  ←→  Open WebUI (port 3000)           │
│   Redis — session memory       Celery — async ingestion      │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool | Version |
|---|---|---|
| LLM runtime | [Ollama](https://ollama.com) | latest |
| LLM (generation) | `llama3.1:8b` | Meta / Apache 2.0 |
| Embeddings | `nomic-embed-text` | Apache 2.0 |
| Vector store | [Qdrant](https://qdrant.tech) | latest |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache 2.0 |
| BM25 | `rank-bm25` | Apache 2.0 |
| PDF parsing | `pymupdf` + `pdfplumber` | AGPL / MIT |
| OCR | `tesseract` + `pytesseract` | Apache 2.0 |
| API framework | [FastAPI](https://fastapi.tiangolo.com) | MIT |
| Chat UI | [Open WebUI](https://github.com/open-webui/open-webui) | MIT |
| Task queue | [Celery](https://docs.celeryq.dev) + Redis | BSD / BSD |
| Database | PostgreSQL | PostgreSQL License |
| Containerisation | Docker Compose | Apache 2.0 |

All components are open source and free to use. No external API keys required.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | v4.x+ with WSL 2 backend (Windows) |
| [Ollama](https://ollama.com/download/windows) | Runs natively on Windows for GPU access |
| 16 GB RAM | Recommended; 8 GB minimum |
| 8 GB+ VRAM | For `llama3.1:8b`; use `llama3.2:3b` on 4 GB VRAM |
| 20 GB disk | Docker images (~4 GB) + Ollama models (~5.2 GB) |

---

## Quick Start

### 1. Clone and configure

```powershell
cd c:\Aparna\rag-chatbot
Copy-Item .env.example .env
# Edit .env if you need to change any defaults
```

### 2. Pull Ollama models

Install Ollama from https://ollama.com/download/windows, then:

```powershell
.\scripts\pull_models.ps1
```

This pulls `llama3.1:8b` (~4.7 GB) and `nomic-embed-text` (~274 MB).

> **Low VRAM?** Edit `.env` and set `LLM_MODEL=llama3.2:3b`, then run `ollama pull llama3.2:3b` instead.

### 3. Start all services

```powershell
docker compose up -d
```

First run pulls Docker images (~2–3 GB total) and builds the API image. Allow 5–10 minutes.

### 4. Verify everything is healthy

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected response:
```json
{ "api": "ok", "ollama": "ok", "qdrant": "ok" }
```

### 5. Open the chat UI

Navigate to **http://localhost:3000** in your browser.

On first launch, Open WebUI asks you to create an admin account — this is local only.

### 6. Ingest a document

```powershell
# Upload a PDF or text file
Invoke-RestMethod -Uri "http://localhost:8000/ingest" `
  -Method Post `
  -Form @{ file = Get-Item "path\to\your\document.pdf" }
```

Poll for completion:
```powershell
Invoke-RestMethod "http://localhost:8000/ingest/{task_id}"
```

### 7. Ask a question

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{ "message": "What is the main topic of the document?" }'
```

---

## Configuration

All settings live in `.env`. Copy `.env.example` to get started.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API URL (Docker reaches host via this address) |
| `LLM_MODEL` | `llama3.1:8b` | Ollama model used for answer generation |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model used for embeddings |
| `EMBED_DIM` | `768` | Embedding vector dimension (must match model) |
| `QDRANT_HOST` | `qdrant` | Qdrant hostname (Docker service name) |
| `QDRANT_COLLECTION` | `rag_documents` | Qdrant collection name |
| `POSTGRES_URL` | `postgresql://raguser:ragpassword@postgres:5432/ragdb` | PostgreSQL connection string |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K_RETRIEVE` | `20` | Candidates retrieved from vector + BM25 search each |
| `TOP_N_RERANK` | `5` | Final chunks passed to the LLM after reranking |

### Model size guide

| Available VRAM | Recommended LLM | Notes |
|---|---|---|
| 4 GB | `llama3.2:3b` | Fast, lower quality |
| 8 GB | `llama3.1:8b` | Good balance (default) |
| 16 GB+ | `mistral:7b` | Higher quality, slower |
| CPU only | `llama3.2:3b` (GGUF Q4) | Functional but slow |

---

## API Reference

All endpoints are served at `http://localhost:8000`. Interactive docs at **http://localhost:8000/docs**.

### `GET /health`

Returns connectivity status for Ollama and Qdrant.

**Response:**
```json
{
  "api": "ok",
  "ollama": "ok",
  "qdrant": "ok"
}
```

---

### `POST /ingest`

Upload a document for ingestion. Processing is asynchronous via Celery.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | file | The document to ingest (`.txt`, `.md`, `.rst`, `.pdf`) |

**Response:**
```json
{
  "task_id": "abc123",
  "filename": "report.pdf",
  "status": "queued"
}
```

---

### `GET /ingest/{task_id}`

Poll the status of an ingestion task.

**Response:**
```json
{
  "task_id": "abc123",
  "status": "SUCCESS",
  "result": {
    "status": "success",
    "chunks_ingested": 47,
    "file": "uploads/report.pdf"
  }
}
```

Possible `status` values: `PENDING`, `STARTED`, `SUCCESS`, `FAILURE`.

---

### `POST /chat`

Send a message and receive a grounded answer with citations.

**Request body:**
```json
{
  "message": "What were the key findings?",
  "session_id": "optional-uuid-for-multi-turn"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | Yes | The user's question |
| `session_id` | string | No | Reuse to maintain conversation history across turns |

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer": "The key findings were... [source: report.pdf, p.4]",
  "sources": [
    { "ref": 1, "file": "report.pdf", "page": 4 },
    { "ref": 2, "file": "report.pdf", "page": 7 }
  ]
}
```

Pass the returned `session_id` back in subsequent requests to continue the conversation. Sessions expire after 1 hour of inactivity.

---

## Project Structure

```
rag-chatbot/
│
├── docker-compose.yml          # All service definitions
├── Dockerfile                  # API + Celery image
├── pyproject.toml              # Python dependencies
├── .env.example                # Environment variable template
├── config.py                   # Centralised settings (pydantic-settings)
│
├── api/                        # FastAPI application
│   ├── main.py                 # App setup, lifespan hooks, CORS
│   └── routes/
│       ├── health.py           # GET /health
│       ├── ingest.py           # POST /ingest, GET /ingest/{task_id}
│       └── chat.py             # POST /chat — full RAG pipeline
│
├── ingestion/                  # Document processing
│   ├── loaders/
│   │   ├── text_loader.py      # .txt, .md, .rst
│   │   └── pdf_loader.py       # pymupdf + pdfplumber + OCR
│   ├── chunker.py              # RecursiveCharacterTextSplitter
│   ├── embedder.py             # Ollama embeddings API
│   ├── pipeline.py             # Orchestrates load → chunk → embed → store
│   └── tasks.py                # Celery async task definition
│
├── retrieval/                  # Search and ranking
│   ├── vector_store.py         # Qdrant client — upsert and search
│   ├── bm25.py                 # In-memory BM25 index (rank-bm25)
│   ├── hybrid.py               # RRF fusion of dense + BM25 results
│   └── reranker.py             # Cross-encoder reranking
│
├── generation/                 # LLM interaction
│   ├── prompts.py              # System prompt and RAG prompt template
│   ├── citation.py             # Formats chunks into cited context blocks
│   └── llm_client.py          # Ollama /api/chat wrapper
│
├── db/                         # Database layer
│   ├── session.py              # SQLAlchemy engine, session factory, init_db
│   └── models.py               # Document and Chunk ORM models
│
├── eval/
│   └── ragas_eval.py           # RAGAS evaluation scaffold (Phase 4)
│
├── scripts/
│   └── pull_models.ps1         # Pull required Ollama models
│
└── uploads/                    # Uploaded files (created at runtime)
```

---

## How It Works

### Ingestion flow

1. File is uploaded via `POST /ingest` and saved to `uploads/`
2. A Celery task picks it up asynchronously
3. The appropriate loader extracts text (page by page for PDFs; OCR if no text layer found)
4. `pdfplumber` separately extracts any embedded tables and serialises them as text rows
5. Text is split into overlapping chunks (512 chars, 64 overlap)
6. Each chunk is embedded using `nomic-embed-text` via Ollama
7. Vectors and payload are stored in Qdrant; metadata stored in PostgreSQL
8. The in-memory BM25 index is rebuilt from the updated Qdrant collection

### Query flow

1. The user's question is embedded with `nomic-embed-text`
2. **Dense search**: top 20 chunks retrieved from Qdrant by cosine similarity
3. **BM25 search**: top 20 chunks retrieved by keyword matching
4. **RRF fusion**: the two ranked lists are merged using Reciprocal Rank Fusion (k=60)
5. **Cross-encoder reranking**: the fused list is rescored by `ms-marco-MiniLM-L-6-v2`, keeping the top 5
6. Context is assembled with numbered citation markers
7. `llama3.1:8b` generates the answer via Ollama's chat API
8. The last 3 conversation turns (6 messages) are included for multi-turn continuity
9. Session history is saved to Redis with a 1-hour TTL

### Why hybrid retrieval?

Dense search finds semantically similar content even when the wording differs, but can miss exact keyword matches. BM25 is strong for specific terms, names, and codes. RRF fusion captures the best of both without needing to tune combination weights.

---

## Supported File Types

| Extension | Loader | Notes |
|---|---|---|
| `.txt` | TextLoader | UTF-8, errors replaced |
| `.md` | TextLoader | Markdown treated as plain text |
| `.rst` | TextLoader | reStructuredText as plain text |
| `.pdf` | PDFLoader | Text extracted page by page; tables extracted separately; OCR used for scanned pages |

### Planned (Phase 3)

| Source | Method |
|---|---|
| `.csv` / `.xlsx` | pandas row serialisation + embeddings |
| PostgreSQL / SQLite | NL→SQL via LlamaIndex `NLSQLTableQueryEngine` |

---

## Development

### Run the API locally (outside Docker)

```powershell
# Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e .

# Point at local services (update .env)
# POSTGRES_URL=postgresql://raguser:ragpassword@localhost:5432/ragdb
# REDIS_URL=redis://localhost:6379/0
# QDRANT_HOST=localhost
# OLLAMA_BASE_URL=http://localhost:11434

uvicorn api.main:app --reload --port 8000
```

### Run the Celery worker locally

```powershell
celery -A ingestion.tasks worker --loglevel=info
```

### Rebuild only the API image after code changes

```powershell
docker compose up -d --build api celery
```

### View live logs

```powershell
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f celery
```

### Stop all services

```powershell
docker compose down
```

### Stop and remove all data volumes (full reset)

```powershell
docker compose down -v
```

---

## Roadmap

### Phase 3 — RDBMS & Tabular Data
- [ ] CSV / Excel ingestion: pandas row serialisation → embeddings
- [ ] PostgreSQL / SQLite: schema introspection + NL→SQL agent
- [ ] `RouterQueryEngine` to route between vector search, SQL, and table engines

### Phase 4 — Evaluation & Hardening
- [ ] RAGAS evaluation suite (faithfulness, context recall, answer relevancy)
- [ ] Build a golden QA dataset from your actual documents
- [ ] Langfuse self-hosted tracing (add to Docker Compose)
- [ ] JWT authentication + rate limiting
- [ ] Quantized model testing (GGUF Q4_K_M) for latency improvement

### Future
- Image ingestion: OpenCLIP embeddings + LLaVA captioning
- Audio ingestion: `faster-whisper` transcription → text pipeline
- Web crawl ingestion: `trafilatura` scraper

---

## Troubleshooting

### Open WebUI doesn't load

Check the container is running and healthy:
```powershell
docker compose ps
```
If `rag_open_webui` shows `Restarting`, check logs:
```powershell
docker logs rag_open_webui --tail 50
```

### API health shows `ollama: unreachable`

Ollama runs natively on Windows, not in Docker. Ensure it is running:
```powershell
# Check the process
Get-Process ollama -ErrorAction SilentlyContinue

# Start it if not running — open from Start Menu or:
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
```

### Docker image pulls fail with EOF errors

This is a known Docker Desktop MTU issue on Windows. The project's setup already applies the fix (`mtu: 1450` in `daemon.json`). If it recurs:
```powershell
# Restart Docker Desktop fully
Get-Process "Docker Desktop" | Stop-Process -Force
Start-Sleep -Seconds 5
Start-Process "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
Start-Sleep -Seconds 30
docker compose up -d
```

### API or Celery containers keep restarting

Check the logs for the error:
```powershell
docker logs rag_api --tail 30
```

Common causes:
- **pydantic validation error**: ensure `extra = "ignore"` is set in `config.py` Settings
- **Ollama unreachable**: Ollama isn't running, or `OLLAMA_BASE_URL` is wrong in `.env`
- **PostgreSQL not ready**: increase the healthcheck `retries` in `docker-compose.yml`

### Slow responses

- Switch to a smaller model: set `LLM_MODEL=llama3.2:3b` in `.env` and run `ollama pull llama3.2:3b`
- Reduce `TOP_K_RETRIEVE` from 20 to 10 in `.env` to speed up retrieval and reranking

### Ingestion task stuck in PENDING

The Celery worker may not be running:
```powershell
docker compose logs celery --tail 20
docker compose restart celery
```
