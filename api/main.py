from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import health, ingest, chat
from db.session import init_db
from retrieval.vector_store import ensure_collection
from retrieval.bm25 import get_bm25_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_collection()
    get_bm25_index().build()
    yield


app = FastAPI(title="RAG Chatbot API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(chat.router, tags=["chat"])
