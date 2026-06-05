from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from api.routes import health, ingest, chat
from api.routes import auth as auth_router
from api.routes import models as models_router
from api.routes import predict as predict_router
from api.routes import openai_compat_chat as openai_compat_router
from db.session import init_db
from retrieval.vector_store import ensure_collection
import generation.model_registry as registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_collection()
    registry.build_registry()
    yield


app = FastAPI(title="RAG Chatbot API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(auth_router.router)
app.include_router(ingest.router, tags=["ingest"])
app.include_router(chat.router, tags=["chat"])
app.include_router(models_router.router)
app.include_router(predict_router.router)
app.include_router(openai_compat_router.router)


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse("frontend/index.html")
