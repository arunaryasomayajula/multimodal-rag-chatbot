from fastapi import APIRouter, Depends
from auth.dependencies import get_current_user
from auth.models import User
import generation.model_registry as registry

router = APIRouter(tags=["models"])


@router.get("/models")
def list_models(current_user: User = Depends(get_current_user)):
    return {"models": registry.list_models()}
