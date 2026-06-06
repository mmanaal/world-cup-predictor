import os

import numpy as np
import pandas as pd

DEFAULT_ELO = 1500.0
K_FACTOR = 32.0

OUTPUT_COLS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "neutral", "outcome", "goal_diff", "is_world_cup",
    "weight", "home_elo", "away_elo", "elo_diff",
]


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float) -> tuple[float, float]:
    ea = expected_score(rating_a, rating_b)
    eb = expected_score(rating_b, rating_a)
    return (
        rating_a + K_FACTOR * (score_a - ea),
        rating_b + K_FACTOR * ((1.0 - score_a) - eb),
    )


def compute_elo_ratings(matches: pd.DataFrame) -> dict[str, float]:
    """Run the full ELO computation over sorted matches; return final ratings."""
    ratings: dict[str, float] = {}
    for _, row in matches.sort_values("date").iterrows():
        home, away = row["home_team"], row["away_team"]
        ra = ratings.get(home, DEFAULT_ELO)
        rb = ratings.get(away, DEFAULT_ELO)
        score_a = {2: 1.0, 1: 0.5, 0: 0.0}.get(int(row["outcome"]), 0.5)
        ratings[home], ratings[away] = update_elo(ra, rb, score_a)
    return ratings


def add_elo_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Add home_elo, away_elo, elo_diff to matches.

    Ratings reflect each team's ELO *before* the match is played,
    updated chronologically using the internal engine.
    """
    ratings: dict[str, float] = {}
    home_elos, away_elos = [], []

    for _, row in matches.sort_values("date").iterrows():
        home, away = row["home_team"], row["away_team"]
        ra = ratings.get(home, DEFAULT_ELO)
        rb = ratings.get(away, DEFAULT_ELO)
        home_elos.append(ra)
        away_elos.append(rb)
        score_a = {2: 1.0, 1: 0.5, 0: 0.0}.get(int(row["outcome"]), 0.5)
        ratings[home], ratings[away] = update_elo(ra, rb, score_a)

    df = matches.sort_values("date").copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df


def main() -> None:
    matches_path = "data/processed/matches_clean.csv"
    out_path = "data/processed/matches_with_elo.csv"

    matches = pd.read_csv(matches_path, parse_dates=["date"])
    result = add_elo_features(matches)

    present_cols = [c for c in OUTPUT_COLS if c in result.columns]
    result = result[present_cols]

    os.makedirs("data/processed", exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"Saved {len(result)} rows to {out_path}")
    print(result[["date", "home_team", "away_team", "home_elo", "away_elo", "elo_diff"]].head())


if __name__ == "__main__":
    main()
