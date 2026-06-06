import sys
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "03_SRC"))

from data.clean_data import clean_dataframe
from features.build_features import build_features
from utils.config import BEST_MODEL_PATH

_model = None
_model_mtime: float | None = None


def load_model():
    global _model, _model_mtime
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("Modèle introuvable. Lancez train_model.py ou optimize_model.py.")
    mtime = BEST_MODEL_PATH.stat().st_mtime
    if _model is None or mtime != _model_mtime:
        _model = joblib.load(BEST_MODEL_PATH)
        _model_mtime = mtime
    return _model


def predict_churn(input_data: dict) -> dict:
    model = load_model()

    df = pd.DataFrame([input_data])
    df = clean_dataframe(df)
    df = build_features(df)

    probability = float(model.predict_proba(df)[0][1])
    prediction = int(probability >= 0.5)

    if probability >= 0.80:
        segment = "Très haut risque"
        action = "Intervention prioritaire"
    elif probability >= 0.60:
        segment = "Risque élevé"
        action = "Contact conseiller"
    elif probability >= 0.30:
        segment = "Risque moyen"
        action = "Campagne personnalisée"
    else:
        segment = "Faible risque"
        action = "Fidélisation classique"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_segment": segment,
        "recommended_action": action,
    }
