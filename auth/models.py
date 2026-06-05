import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Boolean, DateTime, JSON, ForeignKey
from db.session import Base


def _now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=_now)
    preferences = Column(JSON, default=dict)  # {default_model_id, default_embed_model_id}


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False, nullable=False)


class UserCredential(Base):
    __tablename__ = "user_credentials"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    provider = Column(String, nullable=False)   # "ollama" | "vllm" | "openai_compat"
    base_url = Column(String, nullable=False)
    api_key = Column(String, nullable=False)    # Fernet-encrypted
    model_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=_now)
