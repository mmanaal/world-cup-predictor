"""
Head-to-head feature engineering.

For each match, computes H2H stats from the last 5 meetings between the
two teams BEFORE the match date, within a 10-year lookback window.

Competitive filter: if 2+ meetings with tournament_weight >= 2 exist in
the window, use only those; otherwise fall back to all meetings.
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

DATA_PATH = "data/processed/matches_with_features.csv"
MAX_H2H = 5
LOOKBACK_DAYS = 3652          # ~10 years (365.25 * 10)
COMPETITIVE_MIN_TW = 2
MIN_COMPETITIVE = 2

RIVALRY_EXAMPLES = [
    ("Brazil", "Argentina"),
    ("Spain", "Germany"),
    ("England", "France"),
]


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    # Pre-build lookup: canonical pair key → list of records (already date-sorted)
    lookup: dict[tuple, list[dict]] = defaultdict(list)
    for row in df.itertuples(index=False):
        lookup[_key(row.home_team, row.away_team)].append({
            "date": row.date,
            "home": row.home_team,
            "outcome": int(row.outcome),
            "hs": float(row.home_score),
            "as_": float(row.away_score),
            "tw": float(row.tournament_weight),
        })

    lookback = pd.Timedelta(days=LOOKBACK_DAYS)

    h2h_matches_out    = np.zeros(len(df), dtype=np.int8)
    h2h_home_wins_out  = np.zeros(len(df), dtype=np.int8)
    h2h_away_wins_out  = np.zeros(len(df), dtype=np.int8)
    h2h_draws_out      = np.zeros(len(df), dtype=np.int8)
    h2h_home_gf_out    = np.ones(len(df), dtype=np.float32)
    h2h_home_ga_out    = np.ones(len(df), dtype=np.float32)
    h2h_dominance_out  = np.zeros(len(df), dtype=np.float32)

    for idx, row in enumerate(df.itertuples(index=False)):
        match_date = row.date
        home = row.home_team
        away = row.away_team
        cutoff = match_date - lookback

        # All prior meetings within the 10-year window
        all_past = [
            r for r in lookup[_key(home, away)]
            if cutoff <= r["date"] < match_date
        ]

        # Competitive-first filter
        competitive = [r for r in all_past if r["tw"] >= COMPETITIVE_MIN_TW]
        pool = competitive if len(competitive) >= MIN_COMPETITIVE else all_past

        # Most recent MAX_H2H meetings (list is already date-sorted)
        recent = pool[-MAX_H2H:]
        n = len(recent)

        if n == 0:
            # neutral fills already set by np.zeros / np.ones above
            continue

        hw = aw = draws = gf_sum = ga_sum = 0
        for r in recent:
            if r["home"] == home:
                # Same home/away orientation as current match
                gf_sum += r["hs"]
                ga_sum += r["as_"]
                if r["outcome"] == 2:
                    hw += 1
                elif r["outcome"] == 0:
                    aw += 1
                else:
                    draws += 1
            else:
                # Current home team was the away side in this past meeting
                gf_sum += r["as_"]
                ga_sum += r["hs"]
                if r["outcome"] == 0:   # past away win → current home team won
                    hw += 1
                elif r["outcome"] == 2: # past home win → current away team won
                    aw += 1
                else:
                    draws += 1

        h2h_matches_out[idx]   = n
        h2h_home_wins_out[idx] = hw
        h2h_away_wins_out[idx] = aw
        h2h_draws_out[idx]     = draws
        h2h_home_gf_out[idx]   = round(gf_sum / n, 3)
        h2h_home_ga_out[idx]   = round(ga_sum / n, 3)
        h2h_dominance_out[idx] = round((hw - aw) / n, 3)

    df["h2h_matches"]   = h2h_matches_out
    df["h2h_home_wins"] = h2h_home_wins_out
    df["h2h_away_wins"] = h2h_away_wins_out
    df["h2h_draws"]     = h2h_draws_out
    df["h2h_home_gf"]   = h2h_home_gf_out
    df["h2h_home_ga"]   = h2h_home_ga_out
    df["h2h_dominance"] = h2h_dominance_out

    return df


def _rivalry_stats(df: pd.DataFrame, team_a: str, team_b: str) -> pd.Series | None:
    """Return H2H stats for the latest match between two teams."""
    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b))
        | ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    )
    sub = df[mask]
    return sub.sort_values("date").iloc[-1] if not sub.empty else None


def print_rivalry_examples(df: pd.DataFrame):
    h2h_cols = [
        "h2h_matches", "h2h_home_wins", "h2h_away_wins",
        "h2h_draws", "h2h_home_gf", "h2h_home_ga", "h2h_dominance",
    ]
    print("\n" + "=" * 62)
    print("  RIVALRY H2H EXAMPLES  (stats from perspective of home team)")
    print("=" * 62)
    for team_a, team_b in RIVALRY_EXAMPLES:
        row = _rivalry_stats(df, team_a, team_b)
        if row is None:
            print(f"\n  {team_a} vs {team_b}: no matches found")
            continue
        print(
            f"\n  {team_a} / {team_b}"
            f"  — last match {row['date'].date()}"
            f"  (home={row['home_team']}, away={row['away_team']})"
        )
        for col in h2h_cols:
            print(f"    {col:<20} {row[col]}")
    print("=" * 62)


def main():
    print(f"Loading {DATA_PATH} …")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"  {len(df):,} rows loaded")

    print("Computing H2H features …")
    df = compute_h2h(df)

    zero_mask = df["h2h_matches"] == 0
    print(f"\nTotal rows      : {len(df):,}")
    print(f"h2h_matches==0  : {zero_mask.sum():,}  ({100 * zero_mask.mean():.1f}% — neutral fill)")
    print(
        "h2h distribution:",
        dict(df["h2h_matches"].value_counts().sort_index()),
    )

    print_rivalry_examples(df)

    df.to_csv(DATA_PATH, index=False)
    print(f"\nSaved → {DATA_PATH}")


if __name__ == "__main__":
    main()
