from celery import Celery
from config import settings

celery_app = Celery(
    "rag_tasks",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_expires = 3600


@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def ingest_file_task(self, file_path: str, user_id: str | None = None):
    try:
        from ingestion.pipeline import ingest_file
        n_chunks = ingest_file(file_path, user_id=user_id)
        return {"status": "success", "chunks_ingested": n_chunks, "file": file_path}
    except Exception as exc:
        raise self.retry(exc=exc)
