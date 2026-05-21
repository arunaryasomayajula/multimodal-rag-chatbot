import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from ingestion.tasks import celery_app, ingest_file_task
from config import settings

router = APIRouter()

ALLOWED_EXTENSIONS = {".txt", ".md", ".rst", ".pdf"}


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    dest = Path(settings.upload_dir) / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    task = ingest_file_task.delay(str(dest))
    return {"task_id": task.id, "filename": file.filename, "status": "queued"}


@router.get("/ingest/{task_id}")
async def ingest_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result}
