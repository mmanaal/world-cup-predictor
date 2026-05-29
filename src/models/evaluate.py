import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "report": classification_report(
            y_test, y_pred, target_names=["away_win", "draw", "home_win"]
        ),
    }


def print_evaluation(results: dict) -> None:
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Log-loss : {results['log_loss']:.4f}")
    print(results["report"])
