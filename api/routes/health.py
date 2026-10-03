from fastapi import APIRouter
from src.predictor import get_model_info

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/model-info")
def model_info():
    return get_model_info()
