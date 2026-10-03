from fastapi import APIRouter
from api.schemas import FloodInput, FloodPrediction
from src.predictor import predict_flood_probability, risk_level

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=FloodPrediction)
def predict(payload: FloodInput):
    prob = predict_flood_probability(payload.model_dump())
    return FloodPrediction(flood_probability=round(prob, 4), risk_level=risk_level(prob))
