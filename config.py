from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768

    # ── vLLM model instances (each runs in its own container) ──────────────────
    # Qwen3-4B  → port 8001   docker compose --profile qwen up -d
    vllm_qwen_enabled: bool = False
    vllm_qwen_base_url: str = "http://vllm-qwen:8000/v1"
    vllm_qwen_api_key: str = "none"
    vllm_qwen_model: str = "qwen3:4b"

    # Llama-3.2-3B  → port 8002   (requires HF_TOKEN + licence acceptance)
    vllm_llama_enabled: bool = False
    vllm_llama_base_url: str = "http://vllm-llama:8000/v1"
    vllm_llama_api_key: str = "none"
    vllm_llama_model: str = "llama3.2:3b"

    # Mistral-7B  → port 8003
    vllm_mistral_enabled: bool = False
    vllm_mistral_base_url: str = "http://vllm-mistral:8000/v1"
    vllm_mistral_api_key: str = "none"
    vllm_mistral_model: str = "mistral:7b"

    # HuggingFace token — required for gated models (Llama)
    hf_token: str = ""

    # LM Studio / generic OpenAI-compat (e.g. local server on another machine)
    openai_compat_enabled: bool = False
    openai_compat_base_url: str = "http://localhost:1234/v1"
    openai_compat_api_key: str = "none"
    openai_compat_default_model: str = ""

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

    # Auth
    jwt_secret: str = "change-this-jwt-secret-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 30
    credential_encryption_key: str = ""  # Fernet key; generated on first run if empty
    rag_api_key: str = ""  # Static API key for Open WebUI / OpenAI-compat clients


settings = Settings()
