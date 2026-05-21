import json
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import redis as redis_lib
from config import settings
from ingestion.embedder import embed_query
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from generation.llm_client import chat

router = APIRouter()

_redis = redis_lib.from_url(settings.redis_url, decode_responses=True)
_SESSION_TTL = 3600  # seconds


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = _load_history(session_id)

    query_vector = embed_query(req.message)
    candidates = hybrid_search(req.message, query_vector)
    top_chunks = rerank(req.message, candidates)

    if not top_chunks:
        return {
            "session_id": session_id,
            "answer": "No relevant documents found. Please ingest some documents first.",
            "sources": [],
        }

    result = chat(req.message, top_chunks, history)
    _save_history(session_id, req.message, result["answer"])

    return {"session_id": session_id, **result}


def _load_history(session_id: str) -> list[dict]:
    raw = _redis.get(f"session:{session_id}")
    return json.loads(raw) if raw else []


def _save_history(session_id: str, question: str, answer: str):
    history = _load_history(session_id)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    _redis.setex(f"session:{session_id}", _SESSION_TTL, json.dumps(history))
