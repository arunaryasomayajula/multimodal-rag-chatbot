"""
OpenAI-compatible /v1/chat/completions and /v1/models endpoints.
Allows Open WebUI (and any OpenAI SDK client) to use the RAG pipeline.

Auth: static API key from RAG_API_KEY env var. Requests run as a shared
anonymous session (no per-user document isolation — all documents in the
collection are visible).  Set RAG_API_KEY in .env.
"""
import time
import uuid
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from config import settings
from generation.graph import rag_graph
import generation.model_registry as registry

router = APIRouter(prefix="/v1", tags=["openai-compat"])


# ── auth helper ──────────────────────────────────────────────────────────────

def _check_api_key(authorization: str | None):
    if not settings.rag_api_key:
        return  # API key not configured → allow all (dev mode)
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.rag_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── /v1/models ───────────────────────────────────────────────────────────────

@router.get("/models")
def list_models_compat(authorization: str | None = Header(default=None)):
    _check_api_key(authorization)
    models = registry.list_models()
    llm_models = [m for m in models if m["type"] == "llm"]
    return {
        "object": "list",
        "data": [
            {
                "id": m["model_id"],
                "object": "model",
                "created": 0,
                "owned_by": m["provider"],
            }
            for m in llm_models
        ],
    }


# ── /v1/chat/completions ──────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


@router.post("/chat/completions")
def chat_completions(
    req: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
):
    _check_api_key(authorization)

    history = [m.model_dump() for m in req.messages[:-1]]
    query = req.messages[-1].content

    state = {
        "query": query,
        "user_id": None,      # anonymous — sees all documents
        "session_id": str(uuid.uuid4()),
        "model_id": req.model,
        "history": history,
        "active_dataset_id": None,
        "route": "",
        "query_vector": [],
        "candidates": [],
        "chunks": [],
        "answer": "",
        "sources": [],
    }

    try:
        result = rag_graph.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    answer = result["answer"]
    sources = result.get("sources", [])
    if sources:
        citations = "\n\n**Sources:** " + ", ".join(
            f"[{s['ref']}] {s['file']}" + (f" p.{s['page']}" if s.get("page") else "")
            for s in sources
        )
        answer += citations

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if req.stream:
        def _stream():
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model or settings.llm_model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": None}],
            }
            import json
            yield f"data: {json.dumps(chunk)}\n\n"
            done_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": req.model or settings.llm_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or settings.llm_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
