"""
Match-context feature engineering for the World Cup predictor.

Features added:
  fixture_congestion_home / _away  — matches played in the 7 days prior
  is_knockout                      — 1 for World Cup knockout-round matches
  tournament_weight                — 3 / 2 / 1 by competition tier
  rivalry_flag                     — 1 for known rivalry matchups
"""

from __future__ import annotations

import bisect
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Fixture congestion
# ---------------------------------------------------------------------------

def add_fixture_congestion(
    df: pd.DataFrame,
    window_days: int = 7,
) -> pd.DataFrame:
    """
    For each match row, count how many times each team played in the
    `window_days` days strictly before the match date.

    Uses the full dataset as the appearance index — no leakage because
    the window is bounded above by the current match date (exclusive).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Build a sorted list of appearance dates per team (home or away)
    appearances: dict[str, list[pd.Timestamp]] = defaultdict(list)
    for row in df.sort_values("date").itertuples(index=False):
        appearances[row.home_team].append(row.date)
        appearances[row.away_team].append(row.date)

    home_cong: list[int] = []
    away_cong: list[int] = []

    for row in df.itertuples(index=False):
        cutoff = row.date - pd.Timedelta(days=window_days)
        for team, target in ((row.home_team, home_cong), (row.away_team, away_cong)):
            dates = appearances[team]
            lo = bisect.bisect_left(dates, cutoff)
            # bisect_left on match_date → excludes the current match itself
            hi = bisect.bisect_left(dates, row.date)
            target.append(hi - lo)

    df["fixture_congestion_home"] = home_cong
    df["fixture_congestion_away"] = away_cong
    return df


# ---------------------------------------------------------------------------
# 2. is_knockout
# ---------------------------------------------------------------------------

_KNOCKOUT_TERMS = ("quarter", "semi", "final", "round of 16")


def add_is_knockout(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 if the tournament string contains 'World Cup' AND any knockout-round
    keyword (case-insensitive).  Otherwise 0.
    """
    df = df.copy()
    t_lower = df["tournament"].str.lower().fillna("")
    is_wc = t_lower.str.contains("world cup", regex=False)
    is_ko = t_lower.str.contains("|".join(_KNOCKOUT_TERMS), regex=True)
    df["is_knockout"] = (is_wc & is_ko).astype(int)
    return df


# ---------------------------------------------------------------------------
# 3. Tournament weight
# ---------------------------------------------------------------------------

# Tier-2 competitions — checked as substrings (case-insensitive).
# Qualifiers are intentionally excluded: the word "qualif" in the
# tournament name drops it to tier 1.
_TIER2_PATTERNS = (
    "uefa euro",
    "copa am",       # covers Copa América / Copa America
    "africa cup",
    "african cup",
    "afcon",
    "gold cup",
    "asian cup",
    "afc asian cup",
)


def _tournament_weight(name: str) -> int:
    low = name.lower() if isinstance(name, str) else ""
    is_qualifier = "qualif" in low

    if "fifa world cup" in low and not is_qualifier:
        return 3
    if not is_qualifier and any(p in low for p in _TIER2_PATTERNS):
        return 2
    return 1


def add_tournament_weight(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tournament_weight"] = df["tournament"].apply(_tournament_weight)
    return df


# ---------------------------------------------------------------------------
# 4. Rivalry flag
# ---------------------------------------------------------------------------

# Each pair stored as a frozenset so home/away order doesn't matter.
# "USA" normalised to "United States" to match preprocess.py title-casing.
_RIVALRY_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"England",        "Scotland"}),
    frozenset({"Brazil",         "Argentina"}),
    frozenset({"Spain",          "Portugal"}),
    frozenset({"Germany",        "Netherlands"}),
    frozenset({"Italy",          "France"}),
    frozenset({"United States",  "Mexico"}),
    frozenset({"Nigeria",        "Ghana"}),
    frozenset({"Ivory Coast",    "Ghana"}),
    frozenset({"South Korea",    "Japan"}),
    frozenset({"Poland",         "Russia"}),
    frozenset({"Serbia",         "Croatia"}),
    frozenset({"Colombia",       "Ecuador"}),
    frozenset({"Uruguay",        "Argentina"}),
    frozenset({"Chile",          "Peru"}),
    frozenset({"Australia",      "New Zealand"}),
    frozenset({"Hungary",        "Romania"}),
    frozenset({"England",        "Croatia"}),
    frozenset({"Brazil",         "Morocco"}),
    frozenset({"United States",  "Paraguay"}),
    frozenset({"Germany",        "Ivory Coast"}),
    frozenset({"Spain",          "Uruguay"}),
    frozenset({"Germany",        "England"}),
    frozenset({"Germany",        "France"}),
    frozenset({"France",         "Morocco"}),
    frozenset({"Argentina",      "Portugal"}),
})


def add_rivalry_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag 1 when the matchup is a known rivalry, regardless of home/away order."""
    df = df.copy()
    df["rivalry_flag"] = [
        int(frozenset({row.home_team, row.away_team}) in _RIVALRY_PAIRS)
        for row in df.itertuples(index=False)
    ]
    return df


# ---------------------------------------------------------------------------
# Public helper: apply all four in one call
# ---------------------------------------------------------------------------

def add_match_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_fixture_congestion(df)
    df = add_is_knockout(df)
    df = add_tournament_weight(df)
    df = add_rivalry_flag(df)
    return df
