from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_documents"

    # PostgreSQL
    postgres_url: str = "postgresql://raguser:ragpassword@localhost:5432/ragdb"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieve: int = 20
    top_n_rerank: int = 5

    upload_dir: str = "uploads"


settings = Settings()
