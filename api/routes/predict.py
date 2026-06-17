from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional, List
from auth.dependencies import get_current_user, get_db
from auth.models import User
from generation.tabular_predictor import run_prediction
from db.models import PredictionResult, TabularDataset

router = APIRouter(tags=["predict"])


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    dataset_id: str
    target_column: str
    context_rows: int = Field(default=50, ge=1, le=500)
    task_type: str = Field(default="auto", pattern="^(auto|classification|regression|time_series|clustering)$")
    include_metrics: bool = Field(default=True, description="Compute performance metrics")
    include_interpretability: bool = Field(default=False, description="Compute explanations")
    interpretability_methods: Optional[List[str]] = Field(
        default=None,
        description="List of ['feature_importance', 'shap', 'lime']. If None, all are computed."
    )
    n_samples_explain: int = Field(default=100, ge=1, le=1000, description="Number of samples to explain")
    save_result: bool = Field(default=False, description="Save result to session storage")
    notes: Optional[str] = Field(default=None, description="User notes about this prediction")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    task_type: str
    target_column: str
    filename: str
    n_test_rows: int
    n_context_rows: int
    predictions: List
    confidence: List = []
    metrics: Optional[dict] = None
    metrics_visualization: Optional[dict] = None
    interpretability: Optional[dict] = None
    summary: str
    result_id: Optional[str] = None  # ID if saved


class SavedPredictionResponse(BaseModel):
    """Response for saved prediction."""
    id: str
    target_column: str
    task_type: str
    context_rows: int
    created_at: str
    notes: Optional[str] = None
    is_pinned: bool = False


@router.post("/predict", response_model=dict)
def predict(req: PredictRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Run tabular prediction with optional metrics and interpretability.
    
    **Parameters:**
    - `dataset_id`: ID of uploaded dataset
    - `target_column`: Column to predict
    - `context_rows`: Rows for in-context learning (1-500)
    - `task_type`: Type of prediction task
    - `include_metrics`: Whether to compute metrics
    - `include_interpretability`: Whether to compute explanations
    - `interpretability_methods`: Which interpretation methods to use
    - `n_samples_explain`: Number of samples to explain
    - `save_result`: Whether to save to session storage
    - `notes`: Optional user notes
    
    **Returns:**
    Prediction results with optional metrics, visualizations, and interpretability data.
    """
    try:
        result = run_prediction(
            dataset_id=req.dataset_id,
            user_id=current_user.id,
            target_column=req.target_column,
            context_rows=req.context_rows,
            task_type=req.task_type,
            include_metrics=req.include_metrics,
            include_interpretability=req.include_interpretability,
            interpretability_methods=req.interpretability_methods,
            n_samples_explain=req.n_samples_explain,
        )
        
        # Optionally save to database
        result_id = None
        if req.save_result:
            pred_result = PredictionResult(
                user_id=current_user.id,
                dataset_id=req.dataset_id,
                target_column=req.target_column,
                task_type=result.get("task_type"),
                context_rows=req.context_rows,
                predictions=result.get("predictions", []),
                confidence=result.get("confidence", []),
                metrics=result.get("metrics"),
                feature_importance=result.get("interpretability", {}).get("feature_importance"),
                shap_values=result.get("interpretability", {}).get("shap"),
                lime_explanations=result.get("interpretability", {}).get("lime"),
                notes=req.notes,
            )
            db.add(pred_result)
            db.commit()
            db.refresh(pred_result)
            result_id = pred_result.id
            result["result_id"] = result_id
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch")
def predict_batch(
    requests: List[PredictRequest],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Run multiple predictions in batch mode.
    Useful for comparing different target columns or context sizes.
    """
    results = []
    for req in requests:
        try:
            result = run_prediction(
                dataset_id=req.dataset_id,
                user_id=current_user.id,
                target_column=req.target_column,
                context_rows=req.context_rows,
                task_type=req.task_type,
                include_metrics=req.include_metrics,
                include_interpretability=req.include_interpretability,
                interpretability_methods=req.interpretability_methods,
                n_samples_explain=req.n_samples_explain,
            )
            
            # Save if requested
            if req.save_result:
                pred_result = PredictionResult(
                    user_id=current_user.id,
                    dataset_id=req.dataset_id,
                    target_column=req.target_column,
                    task_type=result.get("task_type"),
                    context_rows=req.context_rows,
                    predictions=result.get("predictions", []),
                    confidence=result.get("confidence", []),
                    metrics=result.get("metrics"),
                    feature_importance=result.get("interpretability", {}).get("feature_importance"),
                    shap_values=result.get("interpretability", {}).get("shap"),
                    lime_explanations=result.get("interpretability", {}).get("lime"),
                    notes=req.notes,
                )
                db.add(pred_result)
                db.commit()
                db.refresh(pred_result)
                result["result_id"] = pred_result.id
            
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "target_column": req.target_column})
    
    return {"results": results, "count": len(results)}


@router.get("/predict/datasets")
def list_datasets(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """List tabular datasets available for prediction."""
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


@router.get("/predict/datasets/{dataset_id}")
def get_dataset_info(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get metadata for a specific dataset."""
    row = db.query(TabularDataset).filter(
        TabularDataset.id == dataset_id,
        TabularDataset.user_id == current_user.id,
    ).first()
    
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "id": row.id,
        "filename": row.filename,
        "column_names": row.column_names,
        "row_count": row.row_count,
        "uploaded_at": row.uploaded_at.isoformat() if row.uploaded_at else None,
    }


@router.get("/predict/results")
def list_saved_predictions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    dataset_id: Optional[str] = None,
    limit: int = 50
):
    """
    List saved prediction results.
    
    **Parameters:**
    - `dataset_id`: Filter by dataset (optional)
    - `limit`: Maximum number of results (1-500)
    """
    limit = min(limit, 500)
    query = db.query(PredictionResult).filter(PredictionResult.user_id == current_user.id)
    
    if dataset_id:
        query = query.filter(PredictionResult.dataset_id == dataset_id)
    
    results = query.order_by(PredictionResult.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": r.id,
            "target_column": r.target_column,
            "task_type": r.task_type,
            "context_rows": r.context_rows,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "notes": r.notes,
            "is_pinned": bool(r.is_pinned),
        }
        for r in results
    ]


@router.get("/predict/results/{result_id}")
def get_prediction_result(
    result_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve a saved prediction result."""
    result = db.query(PredictionResult).filter(
        PredictionResult.id == result_id,
        PredictionResult.user_id == current_user.id,
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Prediction result not found")
    
    response = {
        "id": result.id,
        "target_column": result.target_column,
        "task_type": result.task_type,
        "context_rows": result.context_rows,
        "predictions": result.predictions,
        "confidence": result.confidence or [],
        "metrics": result.metrics,
        "created_at": result.created_at.isoformat() if result.created_at else None,
        "notes": result.notes,
        "is_pinned": bool(result.is_pinned),
    }
    
    # Include interpretability if available
    if result.feature_importance:
        response["feature_importance"] = result.feature_importance
    if result.shap_values:
        response["shap_values"] = result.shap_values
    if result.lime_explanations:
        response["lime_explanations"] = result.lime_explanations
    
    return response


@router.put("/predict/results/{result_id}")
def update_prediction_result(
    result_id: str,
    notes: Optional[str] = None,
    is_pinned: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update metadata of a saved prediction."""
    result = db.query(PredictionResult).filter(
        PredictionResult.id == result_id,
        PredictionResult.user_id == current_user.id,
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Prediction result not found")
    
    if notes is not None:
        result.notes = notes
    if is_pinned is not None:
        result.is_pinned = int(is_pinned)
    
    db.commit()
    db.refresh(result)
    
    return {
        "id": result.id,
        "notes": result.notes,
        "is_pinned": bool(result.is_pinned),
    }


@router.delete("/predict/results/{result_id}")
def delete_prediction_result(
    result_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a saved prediction result."""
    result = db.query(PredictionResult).filter(
        PredictionResult.id == result_id,
        PredictionResult.user_id == current_user.id,
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Prediction result not found")
    
    db.delete(result)
    db.commit()
    
    return {"message": "Prediction result deleted", "id": result_id}
