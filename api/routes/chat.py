import json
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import redis as redis_lib
from config import settings
from auth.dependencies import get_current_user
from auth.models import User
from generation.graph import rag_graph

router = APIRouter()

_redis = redis_lib.from_url(settings.redis_url, decode_responses=True)
_SESSION_TTL = 3600


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model_id: str | None = None


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, current_user: User = Depends(get_current_user)):
    session_id = req.session_id or str(uuid.uuid4())
    history = _load_history(current_user.id, session_id)
    active_dataset_id = _get_active_dataset(current_user.id, session_id)

    model_id = req.model_id or (current_user.preferences or {}).get("default_model_id")

    state = {
        "query": req.message,
        "user_id": current_user.id,
        "session_id": session_id,
        "model_id": model_id,
        "history": history,
        "active_dataset_id": active_dataset_id,
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

    _save_history(current_user.id, session_id, req.message, result["answer"])

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "sources": result["sources"],
        "route": result.get("route", "rag"),
    }


def _session_key(user_id: str, session_id: str) -> str:
    return f"session:{user_id}:{session_id}"


def _load_history(user_id: str, session_id: str) -> list[dict]:
    raw = _redis.get(_session_key(user_id, session_id))
    return json.loads(raw) if raw else []


def _save_history(user_id: str, session_id: str, question: str, answer: str):
    key = _session_key(user_id, session_id)
    history = _load_history(user_id, session_id)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    _redis.setex(key, _SESSION_TTL, json.dumps(history))


def _get_active_dataset(user_id: str, session_id: str) -> str | None:
    return _redis.get(f"dataset:{user_id}:{session_id}")


@router.post("/chat/set-dataset")
async def set_active_dataset(
    dataset_id: str,
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """Associate a tabular dataset with the chat session for prediction routing."""
    _redis.setex(f"dataset:{current_user.id}:{session_id}", _SESSION_TTL, dataset_id)
    return {"detail": "Dataset set for session", "dataset_id": dataset_id, "session_id": session_id}
