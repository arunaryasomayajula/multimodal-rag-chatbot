from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.loaders.text_loader import RawChunk
from config import settings

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_raw(raw_chunks: list[RawChunk]) -> list[dict]:
    result = []
    for raw in raw_chunks:
        pieces = _splitter.split_text(raw.content)
        for i, piece in enumerate(pieces):
            result.append({"content": piece, "chunk_index": i, **raw.metadata})
    return result
