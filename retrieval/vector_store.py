import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
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


def upsert_chunks(chunks: list[dict], vectors: list[list[float]], doc_id: str,
                  user_id: str | None = None):
    client = get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={**chunk, "doc_id": doc_id, "user_id": user_id},
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)


def search(query_vector: list[float], top_k: int = 20,
           user_id: str | None = None) -> list[dict]:
    client = get_client()
    query_filter = None
    if user_id:
        query_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
    result = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    return [{"score": h.score, "id": str(h.id), **h.payload} for h in result.points]


def scroll_all_chunks(limit: int = 10_000, user_id: str | None = None) -> list[dict]:
    client = get_client()
    scroll_filter = None
    if user_id:
        scroll_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
    records, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=scroll_filter,
        limit=limit,
        with_payload=True,
    )
    return [{"id": str(r.id), **r.payload} for r in records]
