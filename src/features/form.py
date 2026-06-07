"""
Form feature engineering for the World Cup match outcome predictor.

Provides two APIs:
  - compute_recent_form / add_form_features  (legacy, single points metric)
  - add_rich_form_features                   (full 4-metric, 5-game window)
"""

from collections import defaultdict, deque
import os
from typing import NamedTuple

import pandas as pd

from src.features.match_context import add_match_context_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW = 5

# Neutral priors used when a team has no match history at all
NEUTRAL = {
    "form_pts": 0.5,
    "form_gf": 1.0,
    "form_ga": 1.0,
    "form_gd": 0.0,
}

# Points awarded to a team given the encoded outcome (0=away win, 1=draw, 2=home win)
_HOME_PTS = {2: 3, 1: 1, 0: 0}
_AWAY_PTS = {0: 3, 1: 1, 2: 0}


# ---------------------------------------------------------------------------
# Legacy API (preserved for backward compatibility)
# ---------------------------------------------------------------------------

_POINTS = {
    "home": {2: 3, 1: 1, 0: 0},
    "away": {0: 3, 1: 1, 2: 0},
}


def _team_points(outcome: int, perspective: str) -> int:
    return _POINTS[perspective].get(outcome, 0)


def compute_recent_form(matches: pd.DataFrame, team: str, n: int = 5) -> float:
    team_matches = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ].tail(n)
    if team_matches.empty:
        return 0.0
    total = sum(
        _team_points(int(row["outcome"]), "home" if row["home_team"] == team else "away")
        for _, row in team_matches.iterrows()
    )
    return total / (n * 3)


def add_form_features(matches: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    df = matches.sort_values("date").reset_index(drop=True)
    home_form, away_form = [], []
    for i, row in df.iterrows():
        past = df.iloc[:i]
        home_form.append(compute_recent_form(past, row["home_team"], n))
        away_form.append(compute_recent_form(past, row["away_team"], n))
    df["home_form"] = home_form
    df["away_form"] = away_form
    return df


# ---------------------------------------------------------------------------
# Rich form feature engine
# ---------------------------------------------------------------------------

class _MatchRecord(NamedTuple):
    gf: float   # goals scored by this team
    ga: float   # goals conceded by this team
    pts: int    # points earned (0 / 1 / 3)


def _form_stats(history: deque) -> dict:
    """Aggregate a deque of _MatchRecord into the four normalised metrics."""
    if not history:
        return NEUTRAL.copy()
    n = len(history)
    total_pts = sum(r.pts for r in history)
    total_gf  = sum(r.gf  for r in history)
    total_ga  = sum(r.ga  for r in history)
    return {
        "form_pts": total_pts / (n * 3),       # normalised 0-1
        "form_gf":  total_gf  / n,
        "form_ga":  total_ga  / n,
        "form_gd":  (total_gf - total_ga) / n,
    }


def add_rich_form_features(
    matches: pd.DataFrame,
    window: int = WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add 8 form columns to *matches* (sorted chronologically, no leakage).

    Columns added:
        home_form_pts, home_form_gf, home_form_ga, home_form_gd
        away_form_pts, away_form_gf, away_form_ga, away_form_gd

    Returns
    -------
    df : pd.DataFrame
        Original data with the 8 new columns appended.
    thin_games : pd.DataFrame
        Summary of every (team, match_index) where fewer than `window`
        prior games were available.  Columns: team, match_date, games_available.
    """
    df = matches.sort_values("date").reset_index(drop=True)

    # Rolling history: team -> deque of last `window` _MatchRecord
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    home_stats_list: list[dict] = []
    away_stats_list: list[dict] = []
    thin_rows: list[dict] = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]
        outcome = int(row["outcome"])
        hg = float(row["home_score"])
        ag = float(row["away_score"])

        # --- snapshot BEFORE this match (no leakage) ---
        h_hist = history[home]
        a_hist = history[away]

        h_avail = len(h_hist)
        a_avail = len(a_hist)

        if h_avail < window:
            thin_rows.append({"team": home, "match_date": date, "games_available": h_avail})
        if a_avail < window:
            thin_rows.append({"team": away, "match_date": date, "games_available": a_avail})

        home_stats_list.append(_form_stats(h_hist))
        away_stats_list.append(_form_stats(a_hist))

        # --- update history AFTER recording features ---
        history[home].append(_MatchRecord(gf=hg, ga=ag, pts=_HOME_PTS[outcome]))
        history[away].append(_MatchRecord(gf=ag, ga=hg, pts=_AWAY_PTS[outcome]))

    # Unpack into columns
    for prefix, stats_list in (("home", home_stats_list), ("away", away_stats_list)):
        for metric in ("form_pts", "form_gf", "form_ga", "form_gd"):
            df[f"{prefix}_{metric}"] = [s[metric] for s in stats_list]

    thin_df = (
        pd.DataFrame(thin_rows)
        if thin_rows
        else pd.DataFrame(columns=["team", "match_date", "games_available"])
    )
    # Keep only the *first* occurrence per team (the point where they
    # had zero games) to keep the summary concise
    thin_df = (
        thin_df.sort_values("games_available")
        .drop_duplicates(subset="team", keep="first")
        .sort_values("team")
        .reset_index(drop=True)
    )

    return df, thin_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    in_path  = "data/processed/matches_with_elo.csv"
    out_path = "data/processed/matches_with_features.csv"

    matches = pd.read_csv(in_path, parse_dates=["date"])

    print(f"Loaded {len(matches)} matches from {in_path}")

    print("Computing form features …")
    result, thin_df = add_rich_form_features(matches)

    print("Computing match-context features …")
    result = add_match_context_features(result)

    os.makedirs("data/processed", exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result)} rows × {len(result.columns)} columns to {out_path}")
    print(f"Shape: {result.shape}\n")

    new_cols = [
        "date", "home_team", "away_team",
        "fixture_congestion_home", "fixture_congestion_away",
        "is_knockout", "tournament_weight", "rivalry_flag",
    ]
    print("Sample of new columns (5 rows):")
    print(result[new_cols].head(5).to_string(index=False))

    # Distribution summaries
    print(f"\ntournament_weight value counts:")
    print(result["tournament_weight"].value_counts().sort_index().to_string())
    print(f"\nis_knockout matches:  {result['is_knockout'].sum()}")
    print(f"rivalry_flag matches: {result['rivalry_flag'].sum()}")
    print(f"Max fixture congestion (home): {result['fixture_congestion_home'].max()}")
    print(f"Max fixture congestion (away): {result['fixture_congestion_away'].max()}")

    print(f"\nTeams with fewer than {WINDOW} prior games at their first appearance: "
          f"{len(thin_df['team'].unique())}")
    if not thin_df.empty:
        print(thin_df.to_string(index=False))


if __name__ == "__main__":
    main()
