import joblib
import numpy as np

from src.config import FEATURE_NAMES, MODEL_PATH, PREPROCESSOR_PATH, load_model_card
from src import transformers  # noqa: F401

_model = joblib.load(MODEL_PATH)
_preprocessor = joblib.load(PREPROCESSOR_PATH)


def predict_flood_probability(input_dict: dict) -> float:
    X = np.array([[input_dict[f] for f in FEATURE_NAMES]], dtype=float)
    X_processed = _preprocessor.transform(X)
    pred = _model.predict(X_processed)[0]
    return float(np.clip(pred, 0, 1))


def risk_level(prob: float) -> str:
    if prob < 0.4:
        return "Low Risk"
    elif prob < 0.55:
        return "Moderate Risk"
    elif prob < 0.65:
        return "High Risk"
    return "Severe Risk"


def get_model_info() -> dict:
    return load_model_card()
