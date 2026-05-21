import uuid
from sqlalchemy import Column, String, Integer, Text, DateTime, JSON
from sqlalchemy.sql import func
from db.session import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    modality = Column(String, nullable=False)  # text | pdf
    page_count = Column(Integer)
    chunk_count = Column(Integer)
    ingested_at = Column(DateTime, server_default=func.now())
    doc_metadata = Column(JSON, default={})


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer)
    source_file = Column(String, index=True)
    modality = Column(String)
