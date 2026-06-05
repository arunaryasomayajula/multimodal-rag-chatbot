from rank_bm25 import BM25Okapi
from retrieval.vector_store import scroll_all_chunks


class BM25Index:
    def __init__(self):
        self._corpus: list[dict] = []
        self._index: BM25Okapi | None = None

    def build(self, user_id: str | None = None):
        self._corpus = scroll_all_chunks(user_id=user_id)
        tokenized = [doc["content"].lower().split() for doc in self._corpus]
        if tokenized:
            self._index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if not self._index or not self._corpus:
            return []
        scores = self._index.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"score": float(scores[i]), **self._corpus[i]}
            for i in top_indices
            if scores[i] > 0
        ]


# Per-user BM25 indices keyed by user_id; None key = unauthenticated/global
_indices: dict[str | None, BM25Index] = {}


def get_bm25_index(user_id: str | None = None) -> BM25Index:
    if user_id not in _indices:
        _indices[user_id] = BM25Index()
    return _indices[user_id]


def rebuild_index(user_id: str | None = None):
    idx = get_bm25_index(user_id)
    idx.build(user_id=user_id)
