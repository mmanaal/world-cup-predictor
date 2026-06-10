"""
Fetch international match data from FBref via the soccerdata library.

IMPORTANT — FBref data availability for international competitions:
  soccerdata's FBref reader supports INT-World Cup and INT-European
  Championship schedules (dates, teams, scores, rounds) but does NOT
  expose xG columns for these competitions — FBref only tracks xG in
  its domestic-club match-log tables, not in the international
  competition tables it publishes.

  As a result this script fetches all available match metadata and
  adds explicit NaN xG columns so that xg_features.py can merge on
  date + team and apply neutral fills where values are absent.
  If FBref ever exposes xG for international fixtures the only change
  needed is to populate the xg_home / xg_away columns here.
"""

from __future__ import annotations

import logging
import os
import re

import pandas as pd
import soccerdata as sd

# Suppress soccerdata's chatty INFO output
logging.getLogger("soccerdata").setLevel(logging.WARNING)

RAW_OUT = "data/raw/fbref_xg.csv"

# Competitions and seasons to fetch.
# Season labels used by FBref: 4-digit year of the tournament.
COMPETITIONS: list[tuple[str, list[int]]] = [
    ("INT-World Cup",              [2018, 2022]),
    ("INT-European Championship",  [2020, 2024]),  # Euro 2020 was played in 2021
]


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def _parse_score(score: str) -> tuple[float, float]:
    """
    Parse FBref score string e.g. '2–1' → (2.0, 1.0).
    Returns (NaN, NaN) if the match wasn't played / score missing.
    """
    if not isinstance(score, str):
        return float("nan"), float("nan")
    m = re.match(r"(\d+)\D+(\d+)", score)
    if m:
        return float(m.group(1)), float(m.group(2))
    return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_fbref_international(
    competitions: list[tuple[str, list[int]]] = COMPETITIONS,
    no_cache: bool = False,
    force_cache: bool = False,
) -> pd.DataFrame:
    """
    Download schedules for the given competition/season pairs.

    Returns a DataFrame with columns:
        league, season, date, home_team, away_team,
        home_score, away_score,
        xg_home, xg_away    ← always NaN for international competitions
    """
    frames: list[pd.DataFrame] = []

    for league, seasons in competitions:
        for season in seasons:
            print(f"  Fetching {league} {season} …", end=" ", flush=True)
            try:
                fb = sd.FBref(leagues=league, seasons=season, no_cache=no_cache)
                sched = fb.read_schedule(force_cache=force_cache).reset_index()

                # Parse scores
                home_scores, away_scores = zip(
                    *sched["score"].apply(_parse_score)
                ) if len(sched) else ([], [])

                frame = pd.DataFrame(
                    {
                        "league":      sched.get("league", league),
                        "season":      sched.get("season", season),
                        "date":        pd.to_datetime(sched["date"], errors="coerce"),
                        "home_team":   sched["home_team"],
                        "away_team":   sched["away_team"],
                        "home_score":  list(home_scores),
                        "away_score":  list(away_scores),
                        # xG not available via soccerdata for international competitions
                        "xg_home":     float("nan"),
                        "xg_away":     float("nan"),
                    }
                )
                # Drop rows with no date (future/unscheduled fixtures)
                frame = frame.dropna(subset=["date"])
                frames.append(frame)
                print(f"{len(frame)} matches.")

            except Exception as exc:
                print(f"FAILED — {exc}")

    if not frames:
        raise RuntimeError("No FBref data could be fetched for any competition/season.")

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date", "home_team", "away_team"])
        .sort_values(["date", "league"])
        .reset_index(drop=True)
    )
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print("Fetching FBref international match data …")
    df = fetch_fbref_international()

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(RAW_OUT, index=False)
    print(f"\nSaved {len(df)} rows to {RAW_OUT}")

    print("\nColumns:")
    print(" ", df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))

    print(f"\nxg_home non-null: {df['xg_home'].notna().sum()} / {len(df)}")
    print(
        "Note: xG is not published by FBref for international competitions.\n"
        "      xg_home / xg_away will be filled with neutral values (1.0) by\n"
        "      xg_features.py when building matches_with_features.csv."
    )


if __name__ == "__main__":
    main()
