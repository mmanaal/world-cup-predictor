import joblib
import pandas as pd

from src.models.train import FEATURE_COLS


def load_model(model_path: str = "models/xgb_model.pkl"):
    return joblib.load(model_path)


def predict_outcome(model, match_features: dict) -> dict[str, float]:
    X = pd.DataFrame([match_features])[FEATURE_COLS].fillna(0.0)
    probs = model.predict_proba(X)[0]
    return {
        "away_win": float(probs[0]),
        "draw": float(probs[1]),
        "home_win": float(probs[2]),
    }


def predict_batch(model, matches: pd.DataFrame) -> pd.DataFrame:
    X = matches[FEATURE_COLS].fillna(0.0)
    probs = model.predict_proba(X)
    return pd.DataFrame(probs, columns=["away_win_prob", "draw_prob", "home_win_prob"])
