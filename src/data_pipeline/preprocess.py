import numpy as np
import pandas as pd
import os
import sys 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_pipeline.fetch_historical import load_kaggle_results


def drop_incomplete_matches(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


def encode_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """0 = away win, 1 = draw, 2 = home win."""
    df = df.copy()
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    df["outcome"] = pd.array(np.select(conditions, [2, 1, 0], default=-1), dtype="Int8")
    return df

def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_team"] = df["home_team"].str.strip().str.title()
    df["away_team"] = df["away_team"].str.strip().str.title()
    return df

DECAY_HALF_LIFE_DAYS = 5 * 365.25  # 5-year half-life


def apply_decay_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Exponential decay weight: w = 2^(-(ref - date) / half_life). Reference is today."""
    df = df.copy()
    reference = pd.Timestamp.today().normalize()
    days_ago = (reference - pd.to_datetime(df["date"])).dt.days.clip(lower=0)
    df["weight"] = np.power(2.0, -days_ago / DECAY_HALF_LIFE_DAYS)
    return df


def main():
    out_path = "data/processed/matches_clean.csv"

    df = load_kaggle_results()

    # Drop matches before 2010 — ELO and modern football styles diverge sharply earlier
    df = df[df["date"] >= "2010-01-01"].copy()

    # Run pipeline
    df = drop_incomplete_matches(df)
    df = normalize_team_names(df)
    df = encode_outcome(df)

    # Add useful columns
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["is_world_cup"] = df["tournament"].str.contains("FIFA World Cup", na=False)
    df = apply_decay_weights(df)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} matches to {out_path}")
    print(f"\nOutcome distribution:")
    print(df["outcome"].value_counts().rename({2: "Home win", 1: "Draw", 0: "Away win"}))
    print(f"\nWorld Cup matches: {df['is_world_cup'].sum()}")
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")

if __name__ == "__main__":
    main()