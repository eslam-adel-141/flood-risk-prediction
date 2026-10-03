import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
MODEL_CARD_PATH = MODEL_DIR / "model_card.json"


def load_model_card() -> dict:
    with open(MODEL_CARD_PATH, "r") as f:
        return json.load(f)


FEATURE_NAMES = load_model_card()["feature_names"]
