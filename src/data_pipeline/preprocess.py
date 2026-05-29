import numpy as np
import pandas as pd
import os

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
    df["outcome"] = np.select(conditions, [2, 1, 0], default=np.nan).astype("Int8")
    return df


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_team"] = df["home_team"].str.strip().str.title()
    df["away_team"] = df["away_team"].str.strip().str.title()
    return df

def main():
    out_path = "data/processed/matches_clean.csv"

    df = load_kaggle_results()

    # Filter to post-1990 — older data less relevant to modern football
    df = df[df["date"] >= "1990-01-01"].copy()

    # Run pipeline
    df = drop_incomplete_matches(df)
    df = normalize_team_names(df)
    df = encode_outcome(df)

    # Add useful columns
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["is_world_cup"] = df["tournament"].str.contains("FIFA World Cup", na=False)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} matches to {out_path}")
    print(f"\nOutcome distribution:")
    print(df["outcome"].value_counts().rename({2: "Home win", 1: "Draw", 0: "Away win"}))
    print(f"\nWorld Cup matches: {df['is_world_cup'].sum()}")
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")

if __name__ == "__main__":
    main()