from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from auth.dependencies import get_current_user, get_db
from auth.models import User
from generation.tabular_predictor import run_prediction

router = APIRouter(tags=["predict"])


class PredictRequest(BaseModel):
    dataset_id: str
    target_column: str
    context_rows: int = 50
    task_type: str = "auto"   # "auto" | "classify" | "regress"


@router.post("/predict")
def predict(req: PredictRequest, current_user: User = Depends(get_current_user)):
    try:
        result = run_prediction(
            dataset_id=req.dataset_id,
            user_id=current_user.id,
            target_column=req.target_column,
            context_rows=req.context_rows,
            task_type=req.task_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return result


@router.get("/predict/datasets")
def list_datasets(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    from db.models import TabularDataset
    rows = db.query(TabularDataset).filter(TabularDataset.user_id == current_user.id).all()
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "column_names": r.column_names,
            "row_count": r.row_count,
            "uploaded_at": r.uploaded_at.isoformat() if r.uploaded_at else None,
        }
        for r in rows
    ]
