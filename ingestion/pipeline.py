from pathlib import Path
from ingestion.loaders.text_loader import load_text
from ingestion.loaders.pdf_loader import load_pdf
from ingestion.chunker import chunk_raw
from ingestion.embedder import embed_texts
from retrieval.vector_store import ensure_collection, upsert_chunks
from retrieval.bm25 import get_bm25_index
from db.session import SessionLocal
from db.models import Document, Chunk

LOADERS = {
    ".txt": load_text,
    ".md": load_text,
    ".rst": load_text,
    ".pdf": load_pdf,
}


def ingest_file(path: str | Path) -> int:
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    ensure_collection()

    raw_chunks = loader(path)
    chunks = chunk_raw(raw_chunks)
    texts = [c["content"] for c in chunks]
    vectors = embed_texts(texts)

    doc_id = _persist_to_postgres(path, chunks)
    upsert_chunks(chunks, vectors, doc_id)
    get_bm25_index().build()

    return len(chunks)


def _persist_to_postgres(path: Path, chunks: list[dict]) -> str:
    import uuid
    doc_id = str(uuid.uuid4())
    modality = chunks[0].get("modality", "text") if chunks else "text"

    with SessionLocal() as db:
        doc = Document(
            id=doc_id,
            filename=path.name,
            modality=modality,
            chunk_count=len(chunks),
        )
        db.add(doc)
        for c in chunks:
            db.add(Chunk(
                doc_id=doc_id,
                content=c["content"],
                chunk_index=c["chunk_index"],
                page_number=c.get("page_number"),
                source_file=c.get("source_file"),
                modality=c.get("modality"),
            ))
        db.commit()

    return doc_id
