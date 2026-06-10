import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Base feature set (no xG columns — neutral fills only)
FEATURE_COLS = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_form_pts",
    "home_form_gf",
    "home_form_ga",
    "home_form_gd",
    "away_form_pts",
    "away_form_gf",
    "away_form_ga",
    "away_form_gd",
    "neutral",
    "tournament_weight",
    "rivalry_flag",
    "is_knockout",
    "fixture_congestion_home",
    "fixture_congestion_away",
    # H2H features
    "h2h_matches",
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "h2h_home_gf",
    "h2h_home_ga",
    "h2h_dominance",
    # xG features (real data from StatsBomb for WC/Euro/AFCON/Copa, neutral fill elsewhere)
    "home_xg_form",
    "away_xg_form",
    "home_xg_overperf",
    "away_xg_overperf",
]

# Reference point from previous best run (H2H included, no xG, threshold 0.25)
PREV_BEST = {
    "name": "Prev best: threshold 0.25 (no xG)",
    "accuracy": 0.5141,
    "log_loss": 0.9050,
    "draw_f1": 0.3728,
}

DRAW_FEATURE_COLS = FEATURE_COLS + ["elo_closeness", "form_closeness"]

DATA_PATH = "data/processed/matches_with_features.csv"
MODEL_DIR = "models"
TEST_START = "2022-11-20"
DRAW_CLASS = 1
DRAW_THRESHOLD = 0.25


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(int)
    return df


def add_draw_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_closeness"] = (df["elo_diff"].abs() < 100).astype(int)
    df["form_closeness"] = (
        (df["home_form_pts"] - df["away_form_pts"]).abs() < 0.15
    ).astype(int)
    return df


def time_split(df: pd.DataFrame):
    train_df = df[df["date"] < TEST_START].copy()
    test_df = df[df["date"] >= TEST_START].copy()
    return train_df, test_df


def build_arrays(df: pd.DataFrame, feat_cols: list):
    X = df[feat_cols].astype(float)
    y = df["outcome"]
    w = df["weight"]
    return X, y, w


def make_xgb() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        verbosity=0,
    )


def apply_draw_threshold(y_proba: np.ndarray, threshold: float = DRAW_THRESHOLD) -> np.ndarray:
    y_pred = np.argmax(y_proba, axis=1)
    y_pred[y_proba[:, DRAW_CLASS] > threshold] = DRAW_CLASS
    return y_pred


def evaluate_preds(name: str, y_test, y_pred, y_proba) -> dict:
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    draw_f1 = f1_score(y_test, y_pred, labels=[DRAW_CLASS], average=None)[0]

    print(f"\n{'='*54}")
    print(f"  {name}")
    print(f"{'='*54}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Log-loss  : {ll:.4f}")
    print(f"  Draw F1   : {draw_f1:.4f}")
    print()
    print(
        classification_report(
            y_test, y_pred, target_names=["Away win", "Draw", "Home win"]
        )
    )
    return {"name": name, "accuracy": acc, "log_loss": ll, "draw_f1": draw_f1}


def save_shap_plot(model: XGBClassifier, X_test: pd.DataFrame, path: str):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)

    # sv is either a list of (n_samples, n_features) arrays (one per class)
    # or a 3D (n_samples, n_features, n_classes) array depending on SHAP version.
    if isinstance(sv, list):
        mean_abs_shap = np.mean(np.abs(np.array(sv)), axis=0)   # (n_samples, n_features)
    else:
        mean_abs_shap = np.abs(sv).mean(axis=-1)                # 3D → 2D

    n_features = X_test.shape[1]
    fig, ax = plt.subplots(figsize=(10, max(6, n_features * 0.42)))
    shap.summary_plot(
        mean_abs_shap,
        X_test,
        plot_type="dot",
        max_display=n_features,   # show every feature
        show=False,
        plot_size=None,           # let our figsize control the size
    )
    ax = plt.gca()
    ax.set_xlabel("Mean |SHAP value| (averaged across all 3 outcome classes)")
    ax.set_title("Feature importance — beeswarm (mean |SHAP|, all classes)", pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train(data_path: str = DATA_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data …")
    df = load_data(data_path)
    train_df, test_df = time_split(df)
    print(f"  Train : {len(train_df):,} rows  (before {TEST_START})")
    print(f"  Test  : {len(test_df):,} rows  (from {TEST_START} — 2022 WC era)")

    X_train, y_train, w_train = build_arrays(train_df, FEATURE_COLS)
    X_test, y_test, _ = build_arrays(test_df, FEATURE_COLS)

    results: list[dict] = []
    # maps approach name → (model, feat_cols, X_test_for_shap)
    trained: dict[str, tuple] = {}

    # ══════════════════════════════════════════════════════════════════════
    # BASELINE — decay weights only, argmax decision
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n[1/4] Baseline …")
    base_xgb = make_xgb()
    base_xgb.fit(
        X_train, y_train, sample_weight=w_train,
        eval_set=[(X_test, y_test)], verbose=False,
    )
    base_proba = base_xgb.predict_proba(X_test)
    base_pred = np.argmax(base_proba, axis=1)
    results.append(evaluate_preds("Baseline", y_test, base_pred, base_proba))
    trained["Baseline"] = (base_xgb, FEATURE_COLS, X_test)

    # ══════════════════════════════════════════════════════════════════════
    # APPROACH 1 — draw sample weight 2x on top of decay weights
    # ══════════════════════════════════════════════════════════════════════
    print("\n[2/4] Approach 1 — draw 2× upweight …")
    w_draw = np.array(w_train, dtype=float)
    w_draw[y_train.values == DRAW_CLASS] *= 2.0

    cw_xgb = make_xgb()
    cw_xgb.fit(
        X_train, y_train, sample_weight=w_draw,
        eval_set=[(X_test, y_test)], verbose=False,
    )
    cw_proba = cw_xgb.predict_proba(X_test)
    cw_pred = np.argmax(cw_proba, axis=1)
    results.append(evaluate_preds("Approach 1: draw 2× upweight", y_test, cw_pred, cw_proba))
    trained["Approach 1: draw 2× upweight"] = (cw_xgb, FEATURE_COLS, X_test)

    # ══════════════════════════════════════════════════════════════════════
    # APPROACH 2 — threshold tuning (draw if P(draw) > 0.25)
    #   Uses baseline model probabilities; only the decision rule changes.
    #   Log-loss is identical to baseline since the proba output is unchanged.
    # ══════════════════════════════════════════════════════════════════════
    print("\n[3/4] Approach 2 — threshold tuning (draw threshold = 0.25) …")
    thresh_pred = apply_draw_threshold(base_proba)
    results.append(
        evaluate_preds("Approach 2: threshold 0.25", y_test, thresh_pred, base_proba)
    )
    trained["Approach 2: threshold 0.25"] = (base_xgb, FEATURE_COLS, X_test)

    # ══════════════════════════════════════════════════════════════════════
    # APPROACH 3 — draw-specific features (elo_closeness, form_closeness)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[4/4] Approach 3 — draw-specific features …")
    df_ext = add_draw_features(df)
    train_ext, test_ext = time_split(df_ext)
    X_train_ext, y_train_ext, w_train_ext = build_arrays(train_ext, DRAW_FEATURE_COLS)
    X_test_ext, y_test_ext, _ = build_arrays(test_ext, DRAW_FEATURE_COLS)

    feat_xgb = make_xgb()
    feat_xgb.fit(
        X_train_ext, y_train_ext, sample_weight=w_train_ext,
        eval_set=[(X_test_ext, y_test_ext)], verbose=False,
    )
    feat_proba = feat_xgb.predict_proba(X_test_ext)
    feat_pred = np.argmax(feat_proba, axis=1)
    results.append(
        evaluate_preds("Approach 3: draw features", y_test_ext, feat_pred, feat_proba)
    )
    trained["Approach 3: draw features"] = (feat_xgb, DRAW_FEATURE_COLS, X_test_ext)

    # ── Logistic Regression baseline ──────────────────────────────────────
    print("\n[LR] Logistic Regression baseline …")
    scaler = StandardScaler()
    lr = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    lr.fit(scaler.fit_transform(X_train), y_train, sample_weight=w_train)
    lr_proba = lr.predict_proba(scaler.transform(X_test))
    lr_pred = np.argmax(lr_proba, axis=1)
    evaluate_preds("Logistic Regression (baseline)", y_test, lr_pred, lr_proba)

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 68)
    print("  SUMMARY  (with xG features vs previous best)")
    print("=" * 68)
    print(f"  {'Approach':<40} {'Acc':>7} {'LogLoss':>8} {'DrawF1':>8}")
    print(f"  {'-'*40} {'-'*7} {'-'*8} {'-'*8}")
    # Reference row
    print(
        f"  {PREV_BEST['name']:<40} {PREV_BEST['accuracy']:>7.4f}"
        f" {PREV_BEST['log_loss']:>8.4f} {PREV_BEST['draw_f1']:>8.4f}  ← reference"
    )
    print(f"  {'·'*40} {'·'*7} {'·'*8} {'·'*8}")
    best_result = max(results, key=lambda r: r["draw_f1"])
    for r in results:
        tag = "  ◀ best" if r["name"] == best_result["name"] else ""
        print(
            f"  {r['name']:<40} {r['accuracy']:>7.4f} {r['log_loss']:>8.4f}"
            f" {r['draw_f1']:>8.4f}{tag}"
        )
    print("=" * 68)

    # ── Save best model ────────────────────────────────────────────────────
    best_model, best_feat_cols, best_X_test = trained[best_result["name"]]
    print(f"\nBest approach: {best_result['name']}  (draw F1 = {best_result['draw_f1']:.4f})")

    model_path = os.path.join(MODEL_DIR, "xgb_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved model       → {model_path}")

    feat_path = os.path.join(MODEL_DIR, "feature_cols.json")
    with open(feat_path, "w") as f:
        json.dump(best_feat_cols, f, indent=2)
    print(f"Saved feature cols → {feat_path}")

    # Save draw threshold so predict.py can read it
    threshold_used = DRAW_THRESHOLD if "threshold" in best_result["name"] else None
    thresh_path = os.path.join(MODEL_DIR, "draw_threshold.json")
    with open(thresh_path, "w") as f:
        json.dump({"draw_threshold": threshold_used}, f)
    print(f"Saved threshold    → {thresh_path}  (None = use argmax)")

    # ── SHAP summary plot for best model ──────────────────────────────────
    print("\nGenerating SHAP summary plot …")
    shap_path = os.path.join(MODEL_DIR, "shap_summary.png")
    save_shap_plot(best_model, best_X_test, shap_path)
    print(f"Saved SHAP plot    → {shap_path}")


if __name__ == "__main__":
    train()
