import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from ingestion.tasks import celery_app, ingest_file_task
from ingestion.loaders.tabular_loader import TABULAR_EXTENSIONS, load_tabular
from db.models import TabularDataset
from db.session import SessionLocal
from auth.dependencies import get_current_user, get_db
from auth.models import User
from config import settings

router = APIRouter()

_RAG_EXTENSIONS = {".txt", ".md", ".rst", ".pdf"}
_ALL_EXTENSIONS = _RAG_EXTENSIONS | TABULAR_EXTENSIONS


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALL_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    dest = Path(settings.upload_dir) / current_user.id / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    if suffix in TABULAR_EXTENSIONS:
        meta = load_tabular(dest)
        ds = TabularDataset(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            filename=dest.name,
            column_names=meta.column_names,
            row_count=meta.row_count,
        )
        # Store file under user-specific path in filename
        ds.filename = str(dest.relative_to(Path(settings.upload_dir)))
        db.add(ds)
        db.commit()
        db.refresh(ds)
        return {
            "type": "tabular",
            "dataset_id": ds.id,
            "filename": file.filename,
            "columns": meta.column_names,
            "row_count": meta.row_count,
        }

    task = ingest_file_task.delay(str(dest), user_id=current_user.id)
    return {"type": "rag", "task_id": task.id, "filename": file.filename, "status": "queued"}


@router.get("/ingest/{task_id}")
async def ingest_status(task_id: str, current_user: User = Depends(get_current_user)):
    result = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result}
