import httpx
from config import settings

_TIMEOUT = 60.0


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    with httpx.Client(base_url=settings.ollama_base_url, timeout=_TIMEOUT) as client:
        for text in texts:
            resp = client.post("/api/embeddings", json={
                "model": settings.embed_model,
                "prompt": text,
            })
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
