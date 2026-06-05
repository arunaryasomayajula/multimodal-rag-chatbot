from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from config import settings


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode(
        {"sub": subject, "exp": expire},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> str:
    """Returns the subject (user id) or raises JWTError."""
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    sub = payload.get("sub")
    if sub is None:
        raise JWTError("Missing subject")
    return sub
