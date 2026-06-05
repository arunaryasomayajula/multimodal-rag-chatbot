from fastapi import APIRouter, Depends, HTTPException, status, Response, Cookie
from pydantic import BaseModel
from sqlalchemy.orm import Session
from auth.dependencies import get_db, get_current_user
from auth.models import User
import auth.service as svc

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class CredentialRequest(BaseModel):
    provider: str
    base_url: str
    api_key: str
    model_id: str


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user = svc.register(db, req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"id": user.id, "email": user.email}


@router.post("/login")
def login(req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    try:
        access_token, refresh_id = svc.login(db, req.email, req.password)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    response.set_cookie(
        key="refresh_token", value=refresh_id,
        httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30,
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/refresh")
def refresh(response: Response, refresh_token: str | None = Cookie(default=None),
            db: Session = Depends(get_db)):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token")
    try:
        access_token = svc.refresh_access_token(db, refresh_token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
def logout(response: Response, refresh_token: str | None = Cookie(default=None),
           db: Session = Depends(get_db)):
    if refresh_token:
        svc.revoke_refresh_token(db, refresh_token)
    response.delete_cookie("refresh_token")
    return {"detail": "Logged out"}


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email, "preferences": current_user.preferences}


# --- Credential management ---

@router.get("/credentials")
def list_credentials(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return svc.list_credentials(db, current_user.id)


@router.post("/credentials", status_code=status.HTTP_201_CREATED)
def upsert_credential(req: CredentialRequest, db: Session = Depends(get_db),
                       current_user: User = Depends(get_current_user)):
    cred = svc.upsert_credential(db, current_user.id, req.provider, req.base_url, req.api_key, req.model_id)
    return {"id": cred.id, "provider": cred.provider, "model_id": cred.model_id}


@router.delete("/credentials/{credential_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_credential(credential_id: str, db: Session = Depends(get_db),
                       current_user: User = Depends(get_current_user)):
    svc.delete_credential(db, current_user.id, credential_id)
