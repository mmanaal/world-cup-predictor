import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURE_COLS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form",
    "away_form",
    "home_squad_rating",
    "away_squad_rating",
    "home_attack_strength",
    "away_attack_strength",
    "home_defence_strength",
    "away_defence_strength",
]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].fillna(0.0)
    y = df["outcome"]
    return X, y


def train(df: pd.DataFrame, model_path: str = "models/xgb_model.pkl") -> XGBClassifier:
    X, y = build_features(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(model, model_path)
    return model
