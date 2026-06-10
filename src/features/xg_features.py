"""
xG feature engineering for the World Cup predictor.

Merges FBref match data (data/raw/fbref_xg.csv) onto the main
feature table, then computes per-team rolling xG statistics.

Columns added:
  home_xg, away_xg            — match-level xG (NaN → neutral-filled)
  home_xg_form, away_xg_form  — rolling 5-game avg xG before this match
  home_xg_overperf            — rolling 5-game avg of (actual goals − xG)
  away_xg_overperf            — same for away side

Neutral fills (used when no FBref data exists for a match):
  home_xg = away_xg = home_xg_form = away_xg_form = 1.0
  home_xg_overperf = away_xg_overperf = 0.0
"""

from __future__ import annotations

import os
from collections import defaultdict, deque

import pandas as pd

WINDOW = 5

NEUTRAL = {
    "home_xg":          1.0,
    "away_xg":          1.0,
    "home_xg_form":     1.0,
    "away_xg_form":     1.0,
    "home_xg_overperf": 0.0,
    "away_xg_overperf": 0.0,
}

# ---------------------------------------------------------------------------
# Team-name normalisation map
# FBref uses different spellings for some national teams.
# Keys are FBref names; values are the canonical names in matches_clean.csv.
# ---------------------------------------------------------------------------

_FBREF_NAME_MAP: dict[str, str] = {
    "IR Iran":          "Iran",
    "Korea Republic":   "South Korea",
    "United States":    "United States",   # same, but explicit
    "Côte d'Ivoire":    "Ivory Coast",
    "DR Congo":         "Dr Congo",
    "Bosnia-Herzegovina": "Bosnia And Herzegovina",
    "Macedonia":        "North Macedonia",
    "Czech Republic":   "Czech Republic",
    "Trinidad & Tobago": "Trinidad And Tobago",
}


def _normalise_name(name: str) -> str:
    return _FBREF_NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Fuzzy merge helper
# ---------------------------------------------------------------------------

def _merge_xg(
    matches: pd.DataFrame,
    fbref: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join fbref onto matches on (date, home_team, away_team).

    Steps:
      1. Exact join on all three keys.
      2. For unmatched rows try a date + normalised-name join.
    """
    fbref = fbref.copy()
    fbref["date"] = pd.to_datetime(fbref["date"])
    fbref["home_team_norm"] = fbref["home_team"].apply(_normalise_name)
    fbref["away_team_norm"] = fbref["away_team"].apply(_normalise_name)

    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])

    # Step 1 — exact join (rename fbref cols to home_xg/away_xg on the way in)
    fbref_exact = fbref[["date", "home_team", "away_team", "xg_home", "xg_away"]].rename(
        columns={"xg_home": "home_xg", "xg_away": "away_xg"}
    )
    result = matches.merge(fbref_exact, on=["date", "home_team", "away_team"], how="left")

    # Step 2 — fallback: re-try unmatched rows using normalised FBref names
    unmatched = result["home_xg"].isna()
    if unmatched.any():
        fbref_norm = fbref[["date", "home_team_norm", "away_team_norm", "xg_home", "xg_away"]].rename(
            columns={"home_team_norm": "home_team", "away_team_norm": "away_team",
                     "xg_home": "home_xg_b", "xg_away": "away_xg_b"}
        )
        result2 = result[unmatched][["date", "home_team", "away_team"]].merge(
            fbref_norm, on=["date", "home_team", "away_team"], how="left"
        )
        result.loc[unmatched, "home_xg"] = result2["home_xg_b"].values
        result.loc[unmatched, "away_xg"] = result2["away_xg_b"].values

    return result


# ---------------------------------------------------------------------------
# Rolling xG stats (no-leakage rolling window)
# ---------------------------------------------------------------------------

def _add_rolling_xg(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute rolling-window stats for both teams using
    only matches that occurred BEFORE the current row.

    Uses a deque-per-team approach identical to form.py so the pass
    is O(n) with no data leakage.
    """
    df = df.sort_values("date").copy()

    # Per-team history: deque of (xg_scored, xg_conceded, goals_scored, xg_scored_for_overperf)
    xg_hist:   dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))

    home_xg_form_col:     list[float] = []
    away_xg_form_col:     list[float] = []
    home_overperf_col:    list[float] = []
    away_overperf_col:    list[float] = []

    for row in df.itertuples(index=False):
        h = row.home_team
        a = row.away_team

        h_hist = xg_hist[h]
        a_hist = xg_hist[a]

        # --- snapshot BEFORE this match ---
        if h_hist:
            h_xg_f     = sum(r[0] for r in h_hist) / len(h_hist)
            h_overperf = sum(r[1] for r in h_hist) / len(h_hist)
        else:
            h_xg_f     = NEUTRAL["home_xg_form"]
            h_overperf = NEUTRAL["home_xg_overperf"]

        if a_hist:
            a_xg_f     = sum(r[0] for r in a_hist) / len(a_hist)
            a_overperf = sum(r[1] for r in a_hist) / len(a_hist)
        else:
            a_xg_f     = NEUTRAL["away_xg_form"]
            a_overperf = NEUTRAL["away_xg_overperf"]

        home_xg_form_col.append(h_xg_f)
        away_xg_form_col.append(a_xg_f)
        home_overperf_col.append(h_overperf)
        away_overperf_col.append(a_overperf)

        # --- update history AFTER recording ---
        hxg = row.home_xg if not pd.isna(row.home_xg) else NEUTRAL["home_xg"]
        axg = row.away_xg if not pd.isna(row.away_xg) else NEUTRAL["away_xg"]
        hg  = float(row.home_score) if not pd.isna(row.home_score) else hxg
        ag  = float(row.away_score) if not pd.isna(row.away_score) else axg

        # (xg_for, overperf_for)
        xg_hist[h].append((hxg, hg - hxg))
        xg_hist[a].append((axg, ag - axg))

    df["home_xg_form"]     = home_xg_form_col
    df["away_xg_form"]     = away_xg_form_col
    df["home_xg_overperf"] = home_overperf_col
    df["away_xg_overperf"] = away_overperf_col

    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def add_xg_features(
    matches: pd.DataFrame,
    fbref: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge FBref xG onto matches, then add rolling xG columns.
    Rows with no FBref data get neutral values for match-level xG;
    the rolling stats are computed over actual (or neutral-filled) values.
    """
    df = _merge_xg(matches, fbref)

    # Fill match-level xG neutrals
    df["home_xg"] = df["home_xg"].fillna(NEUTRAL["home_xg"])
    df["away_xg"] = df["away_xg"].fillna(NEUTRAL["away_xg"])

    df = _add_rolling_xg(df)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    fbref_path    = "data/raw/fbref_xg.csv"
    features_path = "data/processed/matches_with_features.csv"
    out_path      = features_path  # overwrite in-place

    print(f"Loading {features_path} …")
    matches = pd.read_csv(features_path, parse_dates=["date"])

    print(f"Loading {fbref_path} …")
    try:
        fbref = pd.read_csv(fbref_path, parse_dates=["date"])
    except FileNotFoundError:
        print(
            f"  {fbref_path} not found — run src/data_pipeline/fetch_xg.py first.\n"
            "  Falling back to all-neutral xG fills."
        )
        fbref = pd.DataFrame(columns=["date", "home_team", "away_team", "xg_home", "xg_away"])

    print("Merging and computing rolling xG features …")
    result = add_xg_features(matches, fbref)

    # Summary: count rows that got real (non-neutral) xG from the merge
    real_xg      = int(result["home_xg"].ne(NEUTRAL["home_xg"]).sum())
    neutral_count = len(result) - real_xg
    print(f"\n  Matches with real xG data : {real_xg}")
    print(f"  Matches with neutral fills: {neutral_count}")

    os.makedirs("data/processed", exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result)} rows × {len(result.columns)} columns to {out_path}")
    print(f"Shape: {result.shape}")

    xg_cols = ["date", "home_team", "away_team",
               "home_xg", "away_xg",
               "home_xg_form", "away_xg_form",
               "home_xg_overperf", "away_xg_overperf"]
    print("\nSample of xG columns (5 rows):")
    print(result[xg_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
