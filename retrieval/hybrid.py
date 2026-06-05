from retrieval.vector_store import search as dense_search
from retrieval.bm25 import get_bm25_index
from config import settings

_RRF_K = 60


def hybrid_search(query: str, query_vector: list[float],
                  user_id: str | None = None) -> list[dict]:
    dense = dense_search(query_vector, top_k=settings.top_k_retrieve, user_id=user_id)
    bm25 = get_bm25_index(user_id).search(query, top_k=settings.top_k_retrieve)
    return _rrf_fuse(dense, bm25)


def _rrf_fuse(dense: list[dict], bm25: list[dict]) -> list[dict]:
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for rank, item in enumerate(dense):
        key = item.get("id", item["content"][:40])
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunks[key] = item

    for rank, item in enumerate(bm25):
        key = item.get("id", item["content"][:40])
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunks[key] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunks[k] for k, _ in ranked]
