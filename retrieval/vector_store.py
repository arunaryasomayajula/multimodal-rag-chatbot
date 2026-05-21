import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import settings

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


def ensure_collection():
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
        )


def upsert_chunks(chunks: list[dict], vectors: list[list[float]], doc_id: str):
    client = get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={**chunk, "doc_id": doc_id},
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)


def search(query_vector: list[float], top_k: int = 20) -> list[dict]:
    client = get_client()
    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [{"score": h.score, "id": str(h.id), **h.payload} for h in hits]


def scroll_all_chunks(limit: int = 10_000) -> list[dict]:
    client = get_client()
    records, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=limit,
        with_payload=True,
    )
    return [{"id": str(r.id), **r.payload} for r in records]
