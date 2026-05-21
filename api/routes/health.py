from fastapi import APIRouter
import httpx
from config import settings

router = APIRouter()


@router.get("/health")
async def health():
    status = {"api": "ok", "ollama": "unknown", "qdrant": "unknown"}

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            status["ollama"] = "ok" if r.status_code == 200 else "error"
    except Exception:
        status["ollama"] = "unreachable"

    try:
        from retrieval.vector_store import get_client
        get_client().get_collections()
        status["qdrant"] = "ok"
    except Exception:
        status["qdrant"] = "unreachable"

    return status
