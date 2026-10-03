from pydantic import BaseModel, Field, create_model
from src.config import FEATURE_NAMES

FloodInput = create_model(
    "FloodInput",
    **{name: (float, Field(..., ge=0, le=20)) for name in FEATURE_NAMES},
)


class FloodPrediction(BaseModel):
    flood_probability: float
    risk_level: str
