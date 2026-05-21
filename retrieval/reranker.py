from sentence_transformers import CrossEncoder
from config import settings

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_encoder: CrossEncoder | None = None


def get_encoder() -> CrossEncoder:
    global _encoder
    if _encoder is None:
        _encoder = CrossEncoder(_MODEL_NAME)
    return _encoder


def rerank(query: str, chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []
    encoder = get_encoder()
    pairs = [(query, chunk["content"]) for chunk in chunks]
    scores = encoder.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[: settings.top_n_rerank]]
