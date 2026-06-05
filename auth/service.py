import uuid
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from auth.models import User, RefreshToken, UserCredential
from auth.password import hash_password, verify_password
from auth.jwt import create_access_token
from config import settings

try:
    from cryptography.fernet import Fernet
    _fernet = Fernet(settings.credential_encryption_key.encode()) if settings.credential_encryption_key else None
except Exception:
    _fernet = None


def _encrypt(value: str) -> str:
    if _fernet:
        return _fernet.encrypt(value.encode()).decode()
    return value


def _decrypt(value: str) -> str:
    if _fernet:
        return _fernet.decrypt(value.encode()).decode()
    return value


def register(db: Session, email: str, password: str) -> User:
    if db.query(User).filter(User.email == email).first():
        raise ValueError("Email already registered")
    user = User(email=email, hashed_password=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def login(db: Session, email: str, password: str) -> tuple[str, str]:
    """Returns (access_token, refresh_token_id)."""
    user = db.query(User).filter(User.email == email, User.is_active == True).first()
    if not user or not verify_password(password, user.hashed_password):
        raise ValueError("Invalid credentials")

    access_token = create_access_token(user.id)
    refresh = RefreshToken(
        user_id=user.id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days),
    )
    db.add(refresh)
    db.commit()
    return access_token, refresh.id


def refresh_access_token(db: Session, refresh_token_id: str) -> str:
    token = db.query(RefreshToken).filter(
        RefreshToken.id == refresh_token_id,
        RefreshToken.revoked == False,
    ).first()
    if not token or token.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise ValueError("Invalid or expired refresh token")
    return create_access_token(token.user_id)


def revoke_refresh_token(db: Session, refresh_token_id: str):
    token = db.query(RefreshToken).filter(RefreshToken.id == refresh_token_id).first()
    if token:
        token.revoked = True
        db.commit()


def get_user_by_id(db: Session, user_id: str) -> User | None:
    return db.query(User).filter(User.id == user_id, User.is_active == True).first()


def list_credentials(db: Session, user_id: str) -> list[dict]:
    rows = db.query(UserCredential).filter(UserCredential.user_id == user_id).all()
    return [
        {"id": r.id, "provider": r.provider, "base_url": r.base_url, "model_id": r.model_id}
        for r in rows
    ]


def upsert_credential(db: Session, user_id: str, provider: str,
                       base_url: str, api_key: str, model_id: str) -> UserCredential:
    existing = db.query(UserCredential).filter(
        UserCredential.user_id == user_id,
        UserCredential.provider == provider,
    ).first()
    if existing:
        existing.base_url = base_url
        existing.api_key = _encrypt(api_key)
        existing.model_id = model_id
        db.commit()
        return existing
    cred = UserCredential(
        user_id=user_id, provider=provider,
        base_url=base_url, api_key=_encrypt(api_key), model_id=model_id,
    )
    db.add(cred)
    db.commit()
    db.refresh(cred)
    return cred


def delete_credential(db: Session, user_id: str, credential_id: str):
    cred = db.query(UserCredential).filter(
        UserCredential.id == credential_id,
        UserCredential.user_id == user_id,
    ).first()
    if cred:
        db.delete(cred)
        db.commit()
