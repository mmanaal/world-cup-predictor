"""
Match outcome predictor.

Loads the trained XGBoost model and assembles a feature vector for any
matchup by looking up current ELO, form, and H2H stats from the
processed dataset. Applies the saved draw threshold at inference time.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# ── Paths (resolved relative to this file so CWD doesn't matter) ──────────
_ROOT       = Path(__file__).resolve().parents[2]
MODEL_PATH  = _ROOT / "models" / "xgb_model.joblib"
FEAT_PATH   = _ROOT / "models" / "feature_cols.json"
THRESH_PATH = _ROOT / "models" / "draw_threshold.json"
DATA_PATH   = _ROOT / "data" / "processed" / "matches_with_features.csv"

# ── H2H constants (mirror h2h.py) ─────────────────────────────────────────
_H2H_MAX       = 5
_H2H_LOOKBACK  = pd.Timedelta(days=3652)   # ~10 years
_H2H_COMP_TW   = 2
_H2H_MIN_COMP  = 2
_DRAW_CLASS    = 1

OUTCOME_LABELS = {0: "Away win", 1: "Draw", 2: "Home win"}

# ── Lazy-loaded singletons ─────────────────────────────────────────────────
_model         = None
_feature_cols  = None
_draw_threshold = None
_df            = None
_explainer     = None


def _load_artifacts() -> None:
    global _model, _feature_cols, _draw_threshold, _df, _explainer
    if _model is not None:
        return
    _model = joblib.load(MODEL_PATH)
    with open(FEAT_PATH) as f:
        _feature_cols = json.load(f)
    with open(THRESH_PATH) as f:
        _draw_threshold = json.load(f)["draw_threshold"]
    _df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    _df["neutral"] = _df["neutral"].astype(int)
    _explainer = shap.TreeExplainer(_model)


# ── Feature lookup helpers ─────────────────────────────────────────────────

def _latest_row(team: str) -> pd.Series:
    mask = (_df["home_team"] == team) | (_df["away_team"] == team)
    rows = _df[mask]
    if rows.empty:
        raise ValueError(f"Team '{team}' not found in dataset.")
    return rows.sort_values("date").iloc[-1]


def _team_elo(team: str) -> float:
    row = _latest_row(team)
    return float(row["home_elo"] if row["home_team"] == team else row["away_elo"])


def _team_form(team: str) -> dict:
    row = _latest_row(team)
    if row["home_team"] == team:
        prefix = "home"
    else:
        prefix = "away"
    return {
        "form_pts": float(row[f"{prefix}_form_pts"]),
        "form_gf":  float(row[f"{prefix}_form_gf"]),
        "form_ga":  float(row[f"{prefix}_form_ga"]),
        "form_gd":  float(row[f"{prefix}_form_gd"]),
    }


def _rivalry_flag(home: str, away: str) -> int:
    mask = (
        ((_df["home_team"] == home) & (_df["away_team"] == away))
        | ((_df["home_team"] == away) & (_df["away_team"] == home))
    )
    return int((_df.loc[mask, "rivalry_flag"] == 1).any())


def _h2h_stats(home: str, away: str) -> dict:
    as_of  = pd.Timestamp("today").normalize()
    cutoff = as_of - _H2H_LOOKBACK

    mask = (
        ((_df["home_team"] == home) & (_df["away_team"] == away))
        | ((_df["home_team"] == away) & (_df["away_team"] == home))
    )
    past = (
        _df[mask & (_df["date"] < as_of) & (_df["date"] >= cutoff)]
        .sort_values("date")
    )

    competitive = past[past["tournament_weight"] >= _H2H_COMP_TW]
    pool   = competitive if len(competitive) >= _H2H_MIN_COMP else past
    recent = pool.tail(_H2H_MAX)
    n      = len(recent)

    if n == 0:
        return {
            "h2h_matches": 0, "h2h_home_wins": 0, "h2h_away_wins": 0,
            "h2h_draws": 0, "h2h_home_gf": 1.0, "h2h_home_ga": 1.0,
            "h2h_dominance": 0.0,
        }

    hw = aw = draws = gf_sum = ga_sum = 0
    for row in recent.itertuples(index=False):
        if row.home_team == home:
            gf_sum += row.home_score
            ga_sum += row.away_score
            if row.outcome == 2:   hw += 1
            elif row.outcome == 0: aw += 1
            else:                  draws += 1
        else:
            gf_sum += row.away_score
            ga_sum += row.home_score
            if row.outcome == 0:   hw += 1
            elif row.outcome == 2: aw += 1
            else:                  draws += 1

    return {
        "h2h_matches":   n,
        "h2h_home_wins": hw,
        "h2h_away_wins": aw,
        "h2h_draws":     draws,
        "h2h_home_gf":   round(gf_sum / n, 3),
        "h2h_home_ga":   round(ga_sum / n, 3),
        "h2h_dominance": round((hw - aw) / n, 3),
    }


def _top_shap_factors(X: pd.DataFrame, predicted_class: int) -> list[str]:
    sv = _explainer.shap_values(X)
    # sv is either a list of (1, n_features) arrays or a (1, n_features, n_classes) array
    if isinstance(sv, list):
        class_shap = sv[predicted_class][0]
    else:
        class_shap = sv[0, :, predicted_class]
    top3_idx = np.argsort(np.abs(class_shap))[::-1][:3]
    return [_feature_cols[i] for i in top3_idx]


# ── Public API ─────────────────────────────────────────────────────────────

def predict_match(
    home_team: str,
    away_team: str,
    neutral: bool = True,
    tournament_weight: int = 3,
    is_knockout: bool = False,
) -> dict:
    """
    Predict the outcome of a match between two teams.

    Returns a dict with probabilities, predicted outcome, confidence level,
    and the top 3 SHAP features driving the prediction.
    """
    _load_artifacts()

    home_elo  = _team_elo(home_team)
    away_elo  = _team_elo(away_team)
    home_form = _team_form(home_team)
    away_form = _team_form(away_team)
    h2h       = _h2h_stats(home_team, away_team)
    rivalry   = _rivalry_flag(home_team, away_team)

    features = {
        "elo_diff":                home_elo - away_elo,
        "home_elo":                home_elo,
        "away_elo":                away_elo,
        "home_form_pts":           home_form["form_pts"],
        "home_form_gf":            home_form["form_gf"],
        "home_form_ga":            home_form["form_ga"],
        "home_form_gd":            home_form["form_gd"],
        "away_form_pts":           away_form["form_pts"],
        "away_form_gf":            away_form["form_gf"],
        "away_form_ga":            away_form["form_ga"],
        "away_form_gd":            away_form["form_gd"],
        "neutral":                 int(neutral),
        "tournament_weight":       float(tournament_weight),
        "rivalry_flag":            rivalry,
        "is_knockout":             int(is_knockout),
        "fixture_congestion_home": 0,
        "fixture_congestion_away": 0,
        **h2h,
    }

    X = pd.DataFrame([features])[_feature_cols].astype(float)
    proba = _model.predict_proba(X)[0]   # shape (3,): [away_win, draw, home_win]

    # Apply saved draw threshold (same rule used at training time)
    if _draw_threshold is not None and proba[_DRAW_CLASS] > _draw_threshold:
        predicted_class = _DRAW_CLASS
    else:
        predicted_class = int(np.argmax(proba))

    pred_prob = float(proba[predicted_class])
    if pred_prob > 0.5:
        confidence = "High"
    elif pred_prob > 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "home_team":         home_team,
        "away_team":         away_team,
        "home_win_prob":     round(float(proba[2]), 4),
        "draw_prob":         round(float(proba[1]), 4),
        "away_win_prob":     round(float(proba[0]), 4),
        "predicted_outcome": OUTCOME_LABELS[predicted_class],
        "confidence":        confidence,
        "top_factors":       _top_shap_factors(X, predicted_class),
    }


# ── Legacy compatibility ───────────────────────────────────────────────────

def load_model(model_path: str = str(MODEL_PATH)):
    return joblib.load(model_path)


def predict_outcome(model, match_features: dict) -> dict[str, float]:
    with open(FEAT_PATH) as f:
        feat_cols = json.load(f)
    X = pd.DataFrame([match_features])[feat_cols].fillna(0.0)
    probs = model.predict_proba(X)[0]
    return {"away_win": float(probs[0]), "draw": float(probs[1]), "home_win": float(probs[2])}


def predict_batch(model, matches: pd.DataFrame) -> pd.DataFrame:
    with open(FEAT_PATH) as f:
        feat_cols = json.load(f)
    X = matches[feat_cols].fillna(0.0)
    probs = model.predict_proba(X)
    return pd.DataFrame(probs, columns=["away_win_prob", "draw_prob", "home_win_prob"])


# ── CLI demo ───────────────────────────────────────────────────────────────

def main():
    fixtures = [
        ("Argentina", "France",   True, 3, True),
        ("England",   "Spain",    True, 3, False),
        ("Brazil",    "Germany",  True, 3, True),
        ("Morocco",   "Portugal", True, 3, False),
        ("Japan",     "Germany",  True, 3, False),
    ]

    print("\n" + "=" * 62)
    print("  WORLD CUP MATCH PREDICTIONS")
    print("=" * 62)

    for home, away, neutral, tw, knockout in fixtures:
        r = predict_match(home, away, neutral=neutral,
                          tournament_weight=tw, is_knockout=knockout)
        tag = "  [Knockout]" if knockout else ""
        print(f"\n  {r['home_team']} vs {r['away_team']}{tag}")
        print(f"  {'─' * 44}")
        print(f"  Home win  :  {r['home_win_prob']:.1%}")
        print(f"  Draw      :  {r['draw_prob']:.1%}")
        print(f"  Away win  :  {r['away_win_prob']:.1%}")
        print(f"  Prediction:  {r['predicted_outcome']}  ({r['confidence']} confidence)")
        print(f"  Key factors: {', '.join(r['top_factors'])}")

    print("\n" + "=" * 62)


if __name__ == "__main__":
    main()
