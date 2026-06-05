# RAG Chatbot

A fully local, open-source RAG chatbot with multi-model LLM support (Ollama + dedicated vLLM servers for Qwen3, Llama 3.2, and Mistral), zero-shot tabular prediction via TABICLv2, JWT authentication with per-user document isolation, and a custom web interface — no cloud APIs or API keys required.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Services and Ports](#services-and-ports)
- [Prerequisites](#prerequisites)
- [Quick Start — CPU Only (Ollama)](#quick-start--cpu-only-ollama)
- [Quick Start — GPU (vLLM Models)](#quick-start--gpu-vllm-models)
- [Web Interface](#web-interface)
- [Authentication](#authentication)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Supported File Types](#supported-file-types)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## Features

| Capability | Detail |
|---|---|
| RAG pipeline | Hybrid dense + BM25 retrieval, RRF fusion, cross-encoder reranking |
| LLM backends | Ollama (CPU/GPU), vLLM Qwen3-4B, Llama 3.2-3B, Mistral 7B |
| Orchestration | LangGraph state graph — RAG vs. tabular conditional routing |
| Tabular prediction | TABICLv2 zero-shot classification / regression on uploaded CSV files |
| Authentication | JWT + bcrypt, httpOnly refresh-token cookie, 30-min / 30-day expiry |
| User isolation | Per-user Qdrant payload filters, BM25 indices, and Redis session keys |
| Web UI | Custom dark-theme SPA at port 8000 with model selector and prediction UI |
| Open WebUI | Full Open WebUI at port 3000, wired to the RAG API as its backend |
| OpenAI-compat API | `/v1/chat/completions` + `/v1/models` for any OpenAI SDK client |
| Document types | PDF (OCR fallback), TXT, MD, RST, CSV, Excel, Parquet |

---

## Architecture

### Query pipeline (LangGraph)

```
  Browser / Open WebUI
        │
        ▼
  POST /chat  (JWT auth)
        │
        ▼
  ┌─────────────────┐
  │  route_intent   │  ← checks query keywords + active dataset
  └────────┬────────┘
           │
  ┌────────┴─────────────────────┐
  │                              │
[rag]                       [tabular]
  │                              │
  ▼                              ▼
retrieve                 tabular_predict
(Qdrant dense            (TABICLv2 in-context
 + BM25, user filter)     learning)
  │
  ▼
rerank
(cross-encoder)
  │
  ▼
generate
(Ollama · vLLM-Qwen · vLLM-Llama · vLLM-Mistral)
  │
  ▼
answer + source citations
```

### Ingestion pipeline

```
  RAG files (.txt .md .rst .pdf)
    → loader → chunker (512 chars, 64 overlap)
    → embed (nomic-embed-text via Ollama)
    → Qdrant (vectors + user_id payload)
    → PostgreSQL (document metadata)
    → BM25 index rebuilt (per user)

  Tabular files (.csv .xlsx .xls .parquet)
    → tabular_loader (column names + row count)
    → PostgreSQL (tabular_datasets table only — skips Qdrant)
```

### Service map

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                         User's browser                          │
  └──────────────────┬──────────────────┬───────────────────────────┘
                     │                  │
              port 8000             port 3000
                     │                  │
          ┌──────────▼──────┐  ┌────────▼─────────┐
          │  FastAPI + UI   │  │   Open WebUI     │
          │  (RAG Chatbot)  │  │  (wired to API)  │
          └──────┬──────────┘  └────────┬─────────┘
                 │                      │ /v1/chat/completions
                 └──────────────────────┘
                           │
          ┌────────────────┼──────────────────────┐
          │                │                      │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌───────────▼──────┐
   │   Qdrant    │  │  PostgreSQL │  │     Redis        │
   │  port 6333  │  │  port 5432  │  │    port 6379     │
   └─────────────┘  └─────────────┘  └──────────────────┘

  Ollama (native host, port 11434)   — CPU/GPU, default LLM

  ─ GPU profile (optional, requires NVIDIA GPU + nvidia-container-toolkit) ─
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  vLLM-Qwen   │  │  vLLM-Llama  │  │ vLLM-Mistral │
  │  port 8001   │  │  port 8002   │  │  port 8003   │
  │  Qwen3-4B    │  │ Llama3.2-3B  │  │  Mistral-7B  │
  └──────────────┘  └──────────────┘  └──────────────┘
```

---

## Tech Stack

| Layer | Tool | Licence |
|---|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) 0.2+ | MIT |
| LLM serving (default) | [Ollama](https://ollama.com) — `llama3.1:8b` | MIT |
| LLM serving (GPU) | [vLLM](https://github.com/vllm-project/vllm) | Apache 2.0 |
| LLM — Qwen | `Qwen/Qwen3-4B` (served as `qwen3:4b`) | Apache 2.0 |
| LLM — Llama | `meta-llama/Llama-3.2-3B-Instruct` (served as `llama3.2:3b`) | Meta Community |
| LLM — Mistral | `mistralai/Mistral-7B-Instruct-v0.3` (served as `mistral:7b`) | Apache 2.0 |
| Embeddings | `nomic-embed-text` via Ollama | Apache 2.0 |
| Tabular prediction | [TABICLv2](https://pypi.org/project/tabicl/) (`tabicl`) | MIT |
| Vector store | [Qdrant](https://qdrant.tech) | Apache 2.0 |
| BM25 | `rank-bm25` | Apache 2.0 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache 2.0 |
| PDF parsing | `pymupdf` + `pdfplumber` + Tesseract OCR | AGPL / MIT |
| API framework | [FastAPI](https://fastapi.tiangolo.com) | MIT |
| Auth | `python-jose` + `bcrypt` | MIT / Apache 2.0 |
| Task queue | [Celery](https://docs.celeryq.dev) + Redis | BSD |
| Database | PostgreSQL 16 | PostgreSQL |
| Chat UI (alternative) | [Open WebUI](https://github.com/open-webui/open-webui) | MIT |

---

## Services and Ports

| Container | Image | Port | Profile | Purpose |
|---|---|---|---|---|
| `rag_api` | custom build | **8000** | always | FastAPI + custom web UI |
| `rag_open_webui` | `ghcr.io/open-webui/open-webui` | **3000** | always | Alternative chat UI |
| `rag_qdrant` | `qdrant/qdrant` | 6333 | always | Vector database |
| `rag_postgres` | `postgres:16-alpine` | 5432 | always | Metadata + user records |
| `rag_redis` | `redis:7-alpine` | 6379 | always | Sessions + Celery broker |
| `rag_celery` | custom build | — | always | Async ingestion worker |
| `rag_vllm_qwen` | `vllm/vllm-openai` | 8001 | `gpu`, `qwen` | Qwen3-4B inference |
| `rag_vllm_llama` | `vllm/vllm-openai` | 8002 | `gpu`, `llama` | Llama 3.2-3B inference |
| `rag_vllm_mistral` | `vllm/vllm-openai` | 8003 | `gpu`, `mistral` | Mistral 7B inference |

Ollama runs **natively on the host** (not in Docker) to get direct GPU access.

---

## Prerequisites

### Always required

| Requirement | Notes |
|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | v4.x+ with WSL 2 backend (Windows) |
| [Ollama](https://ollama.com/download) | Native install on host machine |
| 16 GB RAM | 8 GB minimum |
| 20 GB free disk | Docker images + model weights |

### GPU path (optional)

| Requirement | Notes |
|---|---|
| NVIDIA GPU | 6 GB+ VRAM for Llama, 8 GB+ for Qwen, 14 GB+ for Mistral |
| [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Exposes GPU to Docker containers |
| HuggingFace account | Required for **Llama only** (gated model); free at huggingface.co |

---

## Quick Start — CPU Only (Ollama)

This path runs entirely on CPU with no GPU or HuggingFace account needed.

### 1. Clone and create `.env`

```powershell
git clone <repo-url> rag-chatbot
cd rag-chatbot
Copy-Item .env.example .env
```

### 2. Generate secrets

```powershell
# Add both values to .env
python -c "import secrets; print('JWT_SECRET=' + secrets.token_hex(32))"
python -c "from cryptography.fernet import Fernet; print('CREDENTIAL_ENCRYPTION_KEY=' + Fernet.generate_key().decode())"
```

Open `.env` and replace the placeholder values for `JWT_SECRET` and `CREDENTIAL_ENCRYPTION_KEY`.

### 3. Pull Ollama models

Install Ollama from https://ollama.com/download, then:

```powershell
.\scripts\pull_models.ps1
```

This pulls `llama3.1:8b` (~4.7 GB) and `nomic-embed-text` (~274 MB).

> **Lower VRAM?** Set `LLM_MODEL=llama3.2:3b` in `.env` and run `ollama pull llama3.2:3b` before pulling `nomic-embed-text`.

### 4. Start services

```powershell
docker compose up -d
```

First run builds the API image and pulls Docker images — allow 5–10 minutes.

### 5. Verify health

```powershell
Invoke-RestMethod http://localhost:8000/health
# Expected: { "api": "ok", "ollama": "ok", "qdrant": "ok" }
```

### 6. Open the web UI

Navigate to **http://localhost:8000** and register your account. The model selector shows `llama3.1:8b` via Ollama by default.

---

## Quick Start — GPU (vLLM Models)

Run Qwen3-4B, Llama 3.2-3B, and/or Mistral 7B on a dedicated GPU alongside the Ollama default.

### Prerequisites

1. Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and verify with `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`.
2. For **Llama** only: accept the model licence at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct and create a token at https://huggingface.co/settings/tokens.

### 1. Configure `.env`

Enable the models you want and set the HF token if using Llama:

```
# Enable one or more vLLM models
VLLM_QWEN_ENABLED=true
VLLM_LLAMA_ENABLED=true      # also set HF_TOKEN below
VLLM_MISTRAL_ENABLED=true

# Required for Llama (ignored for Qwen and Mistral)
HF_TOKEN=hf_your_token_here
```

### 2. Start model containers

Use the convenience script — it sets the enabled flags and restarts the API:

```powershell
# Start one model at a time (recommended to conserve VRAM)
.\scripts\start_gpu.ps1 qwen       # Qwen3-4B    ~8 GB VRAM
.\scripts\start_gpu.ps1 llama      # Llama 3.2-3B  ~6 GB VRAM
.\scripts\start_gpu.ps1 mistral    # Mistral 7B  ~14 GB VRAM

# Start all three simultaneously (~28 GB VRAM required)
.\scripts\start_gpu.ps1 all
```

Or use Docker Compose profiles directly:

```powershell
docker compose --profile qwen    up -d   # Qwen only
docker compose --profile llama   up -d   # Llama only
docker compose --profile mistral up -d   # Mistral only
docker compose --profile gpu     up -d   # All three
```

### 3. Monitor first-run download

The first container start downloads model weights from HuggingFace (~2–14 GB per model):

```powershell
docker logs rag_vllm_qwen   -f   # watch Qwen download progress
docker logs rag_vllm_llama  -f   # watch Llama download progress
docker logs rag_vllm_mistral -f  # watch Mistral download progress
```

Wait for `Application startup complete.` in each log before sending requests.

### 4. Select the model in the UI

Open **http://localhost:8000** — the model selector dropdown groups models by backend:

```
🐉 Qwen  (vLLM · GPU)
    └─ 🐉 Qwen3 · 4B

🦙 Llama  (vLLM · GPU)
    └─ 🦙 Llama 3.2 · 3B

🌊 Mistral  (vLLM · GPU)
    └─ 🌊 Mistral · 7B

🖥️  Ollama  (local)
    └─ 🦙 Llama 3.1 · 8B
```

The header badge turns **purple** for GPU-backed models and **indigo** for Ollama.

### VRAM reference

| Model | Container | VRAM (bfloat16) | Tip |
|---|---|---|---|
| Qwen3-4B | `rag_vllm_qwen` | ~8 GB | No HF token needed |
| Llama 3.2-3B | `rag_vllm_llama` | ~6 GB | Requires HF token + licence |
| Mistral 7B | `rag_vllm_mistral` | ~14 GB | Use `--dtype float16` for ~7 GB |
| Llama 3.1 8B | Ollama (host) | ~6 GB | Default; no Docker profile needed |

If only one GPU is available, start models individually rather than all together.

---

## Web Interface

### Custom chat UI — http://localhost:8000

The built-in single-page application, served by FastAPI at the root path.

**Sidebar**
- **Foundation Model** — dropdown listing all registered LLMs, grouped by provider. Select any model; the choice is persisted across page refreshes. Hovering over a selection shows the backend type and VRAM requirement.
- **Upload Document** — drag-and-drop or click to upload PDF / TXT / MD / RST for RAG, or CSV / Excel / Parquet for TABICLv2. RAG files are indexed asynchronously (progress shown inline); tabular files are synchronous.
- **TABICLv2 Datasets** — lists every dataset uploaded by the logged-in user. "Activate" wires the dataset to the current chat session so prediction keywords route to TABICLv2. "Predict" opens a full prediction modal with column selection, task-type override, and a results table.

**Chat area**
- Every assistant message carries a route badge: `🔍 RAG` (green) or `📊 TABICLv2` (purple).
- RAG answers show a collapsible source citations panel with file name and page number.
- Tabular answers show a result card with task type, rows predicted, and accuracy/MAE.
- Animated thinking indicator while the model is generating.
- Suggestion chips on empty state.
- `Enter` to send, `Shift+Enter` for newline.

**New chat** — clears session ID, deactivates active dataset, resets conversation history.

**Register / Sign in** — the login screen has two tabs: "Sign in" for returning users and "Create account" for first-time registration. The API validates and auto-logs-in after successful registration.

### Open WebUI — http://localhost:3000

Open WebUI is configured with two backends simultaneously:
- **Ollama** via `http://host.docker.internal:11434` — direct access to all pulled Ollama models.
- **RAG API** via `http://rag_api:8000/v1` — every message goes through the full RAG pipeline (vector search, hybrid retrieval, reranking) before reaching the LLM.

On first launch, Open WebUI asks for a local admin account (separate from the RAG API accounts).

### API docs — http://localhost:8000/docs

Interactive Swagger UI showing all 17 endpoints with request/response schemas.

---

## Authentication

All API endpoints except `GET /health`, `POST /auth/register`, and `POST /auth/login` require a valid JWT access token.

| Token type | Lifetime | How it travels |
|---|---|---|
| Access token | 30 minutes | `Authorization: Bearer <token>` header |
| Refresh token | 30 days | httpOnly `refresh_token` cookie |

**Per-user document isolation** — every uploaded document, chunk, and tabular dataset is tagged with the uploading user's ID:
- **Qdrant** — all searches apply `Filter(must=[FieldCondition("user_id", ...)])` so users never see each other's vectors.
- **BM25** — each user has an independent in-memory BM25 index.
- **Redis** — session keys are prefixed `session:{user_id}:{session_id}`.
- **PostgreSQL** — `user_id` foreign key on `documents`, `chunks`, and `tabular_datasets`.

### Stored credentials (API key management)

Users can save encrypted provider keys (e.g. a remote vLLM API key) via `POST /auth/credentials`. Keys are stored with Fernet symmetric encryption.

---

## Configuration Reference

All settings live in `.env`. Start from `.env.example`.

### Core / Ollama

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API (host → Docker uses this address) |
| `LLM_MODEL` | `llama3.1:8b` | Default Ollama model for generation |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model (fixed; changing requires Qdrant re-index) |
| `EMBED_DIM` | `768` | Must match the embedding model's output dimension |

### vLLM — Qwen3-4B (port 8001)

| Variable | Default | Description |
|---|---|---|
| `VLLM_QWEN_ENABLED` | `false` | Set `true` to register in model registry |
| `VLLM_QWEN_BASE_URL` | `http://vllm-qwen:8000/v1` | Container-internal URL |
| `VLLM_QWEN_MODEL` | `qwen3:4b` | Served model name (matches `--served-model-name`) |

### vLLM — Llama 3.2-3B (port 8002)

| Variable | Default | Description |
|---|---|---|
| `VLLM_LLAMA_ENABLED` | `false` | Set `true` to register in model registry |
| `VLLM_LLAMA_BASE_URL` | `http://vllm-llama:8000/v1` | Container-internal URL |
| `VLLM_LLAMA_MODEL` | `llama3.2:3b` | Served model name |

### vLLM — Mistral 7B (port 8003)

| Variable | Default | Description |
|---|---|---|
| `VLLM_MISTRAL_ENABLED` | `false` | Set `true` to register in model registry |
| `VLLM_MISTRAL_BASE_URL` | `http://vllm-mistral:8000/v1` | Container-internal URL |
| `VLLM_MISTRAL_MODEL` | `mistral:7b` | Served model name |

### HuggingFace

| Variable | Description |
|---|---|
| `HF_TOKEN` | Required for Llama (gated model). Get yours at https://huggingface.co/settings/tokens. Leave empty for Qwen and Mistral. |

### LM Studio / generic OpenAI-compatible

| Variable | Default | Description |
|---|---|---|
| `OPENAI_COMPAT_ENABLED` | `false` | Enable any `/v1` server as an additional backend |
| `OPENAI_COMPAT_BASE_URL` | `http://host.docker.internal:1234/v1` | LM Studio or remote server URL |
| `OPENAI_COMPAT_API_KEY` | `none` | API key (`none` for local servers) |
| `OPENAI_COMPAT_DEFAULT_MODEL` | _(empty)_ | Model name served by that endpoint |

### Infrastructure

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_URL` | `postgresql://raguser:ragpassword@postgres:5432/ragdb` | PostgreSQL DSN |
| `REDIS_URL` | `redis://redis:6379/0` | Redis URL |
| `QDRANT_HOST` | `qdrant` | Qdrant hostname (Docker service name) |
| `QDRANT_COLLECTION` | `rag_documents` | Qdrant collection |

### Auth

| Variable | Description |
|---|---|
| `JWT_SECRET` | **Required.** Random hex string. Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `JWT_ALGORITHM` | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `30` |
| `CREDENTIAL_ENCRYPTION_KEY` | **Required.** Fernet key. Generate: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `RAG_API_KEY` | Static key for Open WebUI and OpenAI SDK clients hitting `/v1/*` |

### RAG tuning

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K_RETRIEVE` | `20` | Candidates from each retriever (dense + BM25) |
| `TOP_N_RERANK` | `5` | Final chunks passed to the LLM after reranking |

---

## API Reference

Base URL: `http://localhost:8000` — interactive docs at `/docs`.

**Public** (no token): `GET /health`, `POST /auth/register`, `POST /auth/login`  
**All others** require: `Authorization: Bearer <access_token>`

---

### Auth

#### `POST /auth/register`
```json
{ "email": "user@example.com", "password": "secret" }
```
Returns `{ "id": "...", "email": "..." }` (status 201).

#### `POST /auth/login`
```json
{ "email": "user@example.com", "password": "secret" }
```
Returns `{ "access_token": "...", "token_type": "bearer" }` and sets an httpOnly `refresh_token` cookie.

#### `POST /auth/refresh`
Uses the httpOnly cookie to issue a new access token without re-login.

#### `POST /auth/logout`
Revokes the refresh token cookie.

#### `GET /auth/me`
Returns `{ "id", "email", "preferences" }` for the authenticated user.

#### `GET /auth/credentials` · `POST /auth/credentials` · `DELETE /auth/credentials/{id}`
CRUD for encrypted provider API keys.

---

### Models

#### `GET /models`
Returns all LLM and embed models currently in the registry.

```json
{
  "models": [
    { "model_id": "qwen3:4b",       "display_name": "🐉 Qwen3 · 4B",   "provider": "vllm-qwen",   "type": "llm" },
    { "model_id": "llama3.2:3b",    "display_name": "🦙 Llama 3.2 · 3B", "provider": "vllm-llama",  "type": "llm" },
    { "model_id": "mistral:7b",     "display_name": "🌊 Mistral · 7B",  "provider": "vllm-mistral","type": "llm" },
    { "model_id": "llama3.1:8b",    "display_name": "🦙 Llama 3.1 · 8B","provider": "ollama",      "type": "llm" },
    { "model_id": "nomic-embed-text","display_name": "🧲 nomic-embed-text","provider": "ollama",   "type": "embed" }
  ]
}
```

Only models whose container is running appear. Unavailable vLLM endpoints are skipped at startup with a warning log.

---

### Ingest

#### `POST /ingest`
Upload a file. RAG files are queued to Celery; tabular files are processed synchronously.

**Request:** `multipart/form-data` with field `file`.

**Response — RAG file:**
```json
{ "type": "rag", "task_id": "abc123", "filename": "report.pdf", "status": "queued" }
```

**Response — tabular file:**
```json
{
  "type": "tabular",
  "dataset_id": "550e8400-...",
  "filename": "iris.csv",
  "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
  "row_count": 150
}
```

#### `GET /ingest/{task_id}`
Poll RAG task status. Values: `PENDING` · `STARTED` · `SUCCESS` · `FAILURE`.

```json
{ "task_id": "abc123", "status": "SUCCESS", "result": { "chunks_ingested": 47 } }
```

---

### Chat

#### `POST /chat`

```json
{
  "message": "What are the key findings?",
  "session_id": "optional-uuid-for-multi-turn",
  "model_id": "qwen3:4b"
}
```

| Field | Required | Description |
|---|---|---|
| `message` | Yes | User query |
| `session_id` | No | Omit to start a new session; reuse to continue a conversation |
| `model_id` | No | Any model from `GET /models`. Falls back to `LLM_MODEL` in `.env`. |

```json
{
  "session_id": "550e8400-...",
  "answer": "The key findings were... [source: report.pdf, p.4]",
  "sources": [ { "ref": 1, "file": "report.pdf", "page": 4 } ],
  "route": "rag"
}
```

`route` is `"rag"` or `"tabular"`. Sessions expire after 1 hour of inactivity.

#### `POST /chat/set-dataset`
Associate a tabular dataset with the session to enable prediction routing.

```
POST /chat/set-dataset?dataset_id=<uuid>&session_id=<uuid>
```

---

### Tabular prediction

#### `POST /predict`
Run explicit TABICLv2 zero-shot prediction.

```json
{
  "dataset_id": "550e8400-...",
  "target_column": "species",
  "context_rows": 50,
  "task_type": "auto"
}
```

| Field | Default | Description |
|---|---|---|
| `dataset_id` | required | ID from `POST /ingest` |
| `target_column` | required | Column to predict |
| `context_rows` | `50` | In-context learning examples |
| `task_type` | `"auto"` | `"classify"`, `"regress"`, or `"auto"` (≤ 20 unique values → classify) |

```json
{
  "predictions": ["setosa", "versicolor", "virginica"],
  "confidence": [0.97, 0.84, 0.91],
  "metrics": { "accuracy": 0.88 },
  "summary": "TABICLv2 classification on 'iris.csv': predicted 100 rows for column 'species'. Accuracy: 0.88",
  "task_type": "classify",
  "target_column": "species",
  "n_test_rows": 100
}
```

#### `GET /predict/datasets`
List all tabular datasets the current user has uploaded.

---

### OpenAI-compatible (`/v1/*`)

These endpoints let any OpenAI SDK client — including Open WebUI — use the RAG pipeline without custom integration. Authentication uses the static `RAG_API_KEY` from `.env`.

#### `GET /v1/models`
Returns all registered LLM models in OpenAI format.

#### `POST /v1/chat/completions`
Runs the full RAG pipeline (or tabular routing if an active dataset is in the session) and returns an OpenAI-format response. Supports both streaming (`"stream": true`) and non-streaming.

```json
{
  "model": "qwen3:4b",
  "messages": [{ "role": "user", "content": "What are the key findings?" }],
  "stream": false
}
```

---

### `GET /health`

```json
{ "api": "ok", "ollama": "ok", "qdrant": "ok" }
```

---

## Project Structure

```
rag-chatbot/
│
├── docker-compose.yml           # All services: infra + API + three vLLM containers
├── Dockerfile                   # API + Celery image (Python 3.11-slim + Tesseract)
├── pyproject.toml               # Python dependencies
├── config.py                    # Pydantic-settings: all env vars with defaults
├── .env.example                 # Annotated template for .env
│
├── frontend/
│   └── index.html               # Custom SPA — served at GET /
│
├── api/
│   ├── main.py                  # FastAPI app, lifespan, router registration
│   └── routes/
│       ├── health.py            # GET /health
│       ├── auth.py              # POST /auth/*
│       ├── ingest.py            # POST /ingest, GET /ingest/{task_id}
│       ├── chat.py              # POST /chat, POST /chat/set-dataset
│       ├── models.py            # GET /models
│       ├── predict.py           # POST /predict, GET /predict/datasets
│       └── openai_compat_chat.py# GET /v1/models, POST /v1/chat/completions
│
├── auth/
│   ├── models.py                # User, RefreshToken, UserCredential SQLAlchemy ORM
│   ├── service.py               # register(), login(), token lifecycle, credential CRUD
│   ├── dependencies.py          # get_current_user() FastAPI dependency
│   ├── jwt.py                   # JWT encode/decode (python-jose)
│   └── password.py              # bcrypt hash + verify
│
├── generation/
│   ├── graph.py                 # LangGraph state graph
│   ├── model_registry.py        # Registry built from config; lazy-init for Celery
│   ├── llm_client.py            # chat() — delegates to registry provider
│   ├── tabular_predictor.py     # TABICLv2 wrapper (auto-detect classify vs. regress)
│   ├── prompts.py               # System + RAG prompt templates
│   ├── citation.py              # Formats source citations
│   └── providers/
│       ├── base.py              # LLMProvider + EmbedProvider Protocol types
│       ├── ollama.py            # /api/chat + /api/embeddings adapter
│       └── openai_compat.py     # /v1/chat/completions + /v1/embeddings adapter
│
├── ingestion/
│   ├── loaders/
│   │   ├── text_loader.py       # .txt .md .rst
│   │   ├── pdf_loader.py        # pymupdf + pdfplumber + Tesseract OCR
│   │   └── tabular_loader.py    # CSV / XLSX / Parquet → TabularMeta
│   ├── chunker.py               # RecursiveCharacterTextSplitter
│   ├── embedder.py              # embed_texts() via model registry
│   ├── pipeline.py              # load → chunk → embed → store (user-scoped)
│   └── tasks.py                 # Celery task: ingest_file_task(path, user_id)
│
├── retrieval/
│   ├── vector_store.py          # Qdrant upsert + query_points (user_id filter)
│   ├── bm25.py                  # Per-user BM25Index dict
│   ├── hybrid.py                # RRF fusion (dense + BM25)
│   └── reranker.py              # Cross-encoder reranking
│
├── db/
│   ├── session.py               # SQLAlchemy engine, SessionLocal, init_db()
│   └── models.py                # Document, Chunk, TabularDataset ORM
│
├── scripts/
│   ├── pull_models.ps1          # Pull llama3.1:8b + nomic-embed-text from Ollama
│   ├── start_gpu.ps1            # Enable + start one or all vLLM containers
│   └── migrate_add_user_id.py   # Safe migration for v0.1 → v0.2 databases
│
└── uploads/                     # Runtime upload storage (gitignored)
    └── {user_id}/
        └── filename.ext
```

---

## How It Works

### RAG document flow

1. `POST /ingest` saves the file to `uploads/{user_id}/filename` and queues a Celery task.
2. The loader extracts text (page by page for PDFs; OCR if no text layer; tables serialised as rows).
3. Text is split into 512-character chunks with 64-character overlap.
4. Each chunk is embedded with `nomic-embed-text` via Ollama.
5. Vectors are upserted to Qdrant with `user_id` in the payload; metadata goes to PostgreSQL.
6. The per-user BM25 index is rebuilt from the user's updated Qdrant slice.

### Tabular dataset flow

1. `POST /ingest` saves the CSV/XLSX/Parquet to `uploads/{user_id}/filename`.
2. `tabular_loader` reads the header row and row count — it does **not** embed anything.
3. A `TabularDataset` record is written to PostgreSQL and the dataset ID returned immediately.
4. At prediction time, TABICLv2 reads the full file, splits into context rows (in-context training set) and test rows, and predicts without fine-tuning.

### Query flow (LangGraph state graph)

1. JWT is validated; `user_id` extracted.
2. Session history is loaded from Redis (`session:{user_id}:{session_id}`).
3. **`route_intent` node**: if the query contains prediction keywords (`predict`, `classify`, `forecast`, `regress`) **and** a dataset is active in the session → `tabular` route. Otherwise → `rag` route.
4. **RAG route**: embed query → `hybrid_search()` (Qdrant dense + BM25, user-filtered, RRF-fused) → cross-encoder rerank (top 5) → LLM generates answer with numbered citations.
5. **Tabular route**: `predict_from_session()` extracts target column from the query → `TabICLClassifier` or `TabICLRegressor` → structured summary returned as the answer.
6. Answer + updated history written back to Redis.

### Model registry

At startup (`lifespan` hook in `api/main.py`) `build_registry()` is called:
- Ollama's `/api/tags` is queried; every listed model is registered.
- Each enabled vLLM endpoint is probed via `/v1/models`; discovered models are registered under `vllm-{name}`. Unreachable endpoints are skipped with a warning — the API starts normally.
- The embed model is always registered pointing to Ollama, regardless of which LLM providers are active.

The Celery worker lazy-initialises the registry on first embed call (since it doesn't run the FastAPI lifespan).

---

## Supported File Types

### RAG (chunked → embedded → Qdrant)

| Extension | Notes |
|---|---|
| `.txt` `.md` `.rst` | Plain-text UTF-8 |
| `.pdf` | Text extracted page-by-page; embedded tables serialised as text; scanned pages run through Tesseract OCR |

### Tabular (stored in PostgreSQL only — no embedding)

| Extension | Notes |
|---|---|
| `.csv` | UTF-8, pandas |
| `.xlsx` `.xls` | openpyxl |
| `.parquet` | pyarrow / fastparquet |

---

## Development

### Run API locally (outside Docker)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .

# In .env, point at local services:
# POSTGRES_URL=postgresql://raguser:ragpassword@localhost:5432/ragdb
# REDIS_URL=redis://localhost:6379/0
# QDRANT_HOST=localhost
# OLLAMA_BASE_URL=http://localhost:11434

uvicorn api.main:app --reload --port 8000
```

### Run Celery locally

```powershell
celery -A ingestion.tasks worker --loglevel=info
```

### Rebuild containers after code changes

```powershell
# Full rebuild with cache-busting for the code layer:
$bust = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
docker compose build --build-arg "CACHEBUST=$bust" api

# Then restart:
docker compose up -d api celery
```

### Run the v0.1 → v0.2 migration (existing databases only)

```powershell
docker exec rag_api python scripts/migrate_add_user_id.py
```

New installations: `init_db()` at startup creates all tables automatically.

### View logs

```powershell
docker compose logs -f                     # all services
docker compose logs -f api                 # FastAPI
docker compose logs -f celery             # ingestion worker
docker logs rag_vllm_qwen   --tail 50 -f  # Qwen3 download / startup
docker logs rag_vllm_llama  --tail 50 -f  # Llama download / startup
docker logs rag_vllm_mistral --tail 50 -f # Mistral download / startup
```

### Stop services

```powershell
docker compose down            # stop all containers
docker compose down -v         # stop + wipe all data volumes (full reset)
```

---

## Troubleshooting

### `401 Unauthorized` on requests

Access tokens expire in 30 minutes. Call `POST /auth/refresh` (with the httpOnly cookie) to get a new token, or sign in again at **http://localhost:8000**.

### `ollama: unreachable` in health check

Ollama runs natively on the host, not in Docker. Verify it is running:

```powershell
Get-Process ollama -ErrorAction SilentlyContinue
# Start if not running:
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
```

### API or Celery container keeps restarting

```powershell
docker logs rag_api --tail 30
```

Common causes:
- `JWT_SECRET` or `CREDENTIAL_ENCRYPTION_KEY` blank or invalid in `.env`.
- PostgreSQL still starting — increase `retries` in the `postgres` healthcheck.
- Ollama unreachable at startup — the registry logs a warning and continues; health endpoint reflects the state.

### vLLM container exits immediately

- `HUGGING_FACE_HUB_TOKEN` not set (required for Llama).
- For Llama: licence not accepted at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct.
- No NVIDIA GPU, or `nvidia-container-toolkit` not installed — verify with `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`.
- Insufficient VRAM — try a smaller model or lower `--gpu-memory-utilization`.

```powershell
docker logs rag_vllm_qwen --tail 50
```

### vLLM model not appearing in the selector

1. Check the `*_ENABLED=true` flag is set in `.env` for that model.
2. Check the container is running: `docker ps | Select-String vllm`.
3. Restart the API to re-probe the endpoint: `docker compose restart api`.

### Ingestion task stuck in PENDING

Celery worker may have stopped:

```powershell
docker compose logs celery --tail 20
docker compose restart celery
```

### Slow RAG responses

- Use a faster model: set `LLM_MODEL=llama3.2:3b` in `.env` and run `ollama pull llama3.2:3b`.
- Reduce candidates: set `TOP_K_RETRIEVE=10` in `.env`.
- For GPU acceleration, start any vLLM container and select it in the model selector.

### Docker image pulls fail (Windows MTU issue)

```powershell
Get-Process "Docker Desktop" | Stop-Process -Force
Start-Sleep -Seconds 10
Start-Process "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
Start-Sleep -Seconds 30
docker compose up -d
```

### Open WebUI doesn't load

```powershell
docker compose ps
docker logs rag_open_webui --tail 50
```

If the RAG API backend shows errors in Open WebUI, verify `RAG_API_KEY` in `.env` matches the `OPENAI_API_KEY` in the Open WebUI service definition in `docker-compose.yml`.
