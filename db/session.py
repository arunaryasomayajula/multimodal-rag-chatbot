from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config import settings

engine = create_engine(settings.postgres_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from db import models  # noqa: F401 — registers Document, Chunk, TabularDataset
    from auth import models as _auth_models  # noqa: F401 — registers User, RefreshToken, UserCredential
    Base.metadata.create_all(bind=engine)
