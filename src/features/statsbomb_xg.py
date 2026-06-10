"""
Replace neutral xG fills with real StatsBomb data.

Pipeline
--------
1. Fetch shot events from 6 international tournaments (314 matches).
2. Aggregate shot_statsbomb_xg per team per match → (date, home_team, away_team, home_xg, away_xg).
3. Normalise StatsBomb team names to match our dataset.
4. Merge onto matches_with_features.csv (exact + swapped-home/away fallback).
5. Recompute home/away xG form and xG over-performance as rolling 5-game averages.
6. Save back to matches_with_features.csv.
"""
import sys
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from statsbombpy import sb

DATA_PATH = "data/processed/matches_with_features.csv"
XG_WINDOW = 5
XG_NEUTRAL = 1.0   # fill value when no history yet (matches original behaviour)

# ── Competition / season pairs to pull ────────────────────────────────────
COMPETITIONS = [
    (43,    3, "FIFA World Cup 2018"),
    (43,  106, "FIFA World Cup 2022"),
    (55,   43, "UEFA Euro 2020"),
    (55,  282, "UEFA Euro 2024"),
    (223, 282, "Copa America 2024"),
    (1267, 107, "AFCON 2023"),
]

# ── StatsBomb → our dataset name map ──────────────────────────────────────
NAME_MAP: dict[str, str] = {
    "Cape Verde Islands": "Cape Verde",
    "Congo DR":           "Dr Congo",
    "Côte d'Ivoire":      "Ivory Coast",
}


def _norm(name: str) -> str:
    return NAME_MAP.get(name, name)


# ── Step 1 + 2: fetch and aggregate xG per match ──────────────────────────

def _xg_for_match(match_id: int, home_team: str, away_team: str) -> dict:
    """Return {match_id, home_xg, away_xg} for one match."""
    events = sb.events(match_id=match_id)
    shots = events[events["type"] == "Shot"]
    home_xg = float(shots.loc[shots["team"] == home_team, "shot_statsbomb_xg"].sum())
    away_xg = float(shots.loc[shots["team"] == away_team, "shot_statsbomb_xg"].sum())
    return {"match_id": match_id, "home_xg": home_xg, "away_xg": away_xg}


def fetch_statsbomb_xg(max_workers: int = 6) -> pd.DataFrame:
    """
    Pull all matches for the 6 competitions, fetch shot events in parallel,
    and return a DataFrame with columns:
        date, home_team, away_team, home_xg, away_xg, competition
    """
    # --- collect all match metadata first ---
    all_matches: list[dict] = []
    for cid, sid, label in COMPETITIONS:
        matches = sb.matches(competition_id=cid, season_id=sid)
        for _, row in matches.iterrows():
            all_matches.append({
                "match_id":  int(row["match_id"]),
                "date":      str(row["match_date"]),
                "home_team": _norm(str(row["home_team"])),
                "away_team": _norm(str(row["away_team"])),
                # keep original SB names for the shot lookup (SB uses those internally)
                "_sb_home":  str(row["home_team"]),
                "_sb_away":  str(row["away_team"]),
                "competition": label,
            })

    total = len(all_matches)
    print(f"  Fetching events for {total} matches across {len(COMPETITIONS)} competitions …")

    # --- fetch events in parallel ---
    xg_lookup: dict[int, dict] = {}

    def _fetch(m: dict) -> dict:
        result = _xg_for_match(m["match_id"], m["_sb_home"], m["_sb_away"])
        return {**m, **result}

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, m): m for m in all_matches}
        for future in as_completed(futures):
            done += 1
            if done % 20 == 0 or done == total:
                print(f"    {done}/{total} matches fetched …", flush=True)
            row = future.result()
            xg_lookup[row["match_id"]] = row

    rows = list(xg_lookup.values())
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "home_team", "away_team", "home_xg", "away_xg", "competition"]]
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Step 3: merge onto main dataset ───────────────────────────────────────

def merge_xg(main: pd.DataFrame, xg: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Join real xG onto main on (date, home_team, away_team).
    Falls back to swapped home/away (neutral-venue artefact).
    Returns (patched_df, n_direct_hits, n_swap_hits).
    """
    main = main.copy()
    main["date"] = pd.to_datetime(main["date"])

    # Build fast lookup: (date, home, away) → (home_xg, away_xg)
    direct: dict[tuple, tuple] = {}
    swapped: dict[tuple, tuple] = {}
    for _, r in xg.iterrows():
        key = (r["date"].date(), r["home_team"], r["away_team"])
        direct[key]  = (r["home_xg"], r["away_xg"])
        swap_key = (r["date"].date(), r["away_team"], r["home_team"])
        swapped[swap_key] = (r["away_xg"], r["home_xg"])   # xG swapped to match perspective

    n_direct = n_swap = 0
    home_xg_out = main["home_xg"].tolist()
    away_xg_out = main["away_xg"].tolist()

    for i, row in enumerate(main.itertuples(index=False)):
        key = (row.date.date(), row.home_team, row.away_team)
        if key in direct:
            home_xg_out[i], away_xg_out[i] = direct[key]
            n_direct += 1
        elif key in swapped:
            home_xg_out[i], away_xg_out[i] = swapped[key]
            n_swap += 1

    main["home_xg"] = home_xg_out
    main["away_xg"] = away_xg_out
    return main, n_direct, n_swap


# ── Step 4: rolling xG features ───────────────────────────────────────────

def compute_xg_rolling(df: pd.DataFrame, window: int = XG_WINDOW) -> pd.DataFrame:
    """
    Recompute home/away xG form and over-performance using an O(n) deque
    over the chronologically sorted dataset (no leakage — snapshot before
    each match, update after).

    Columns written:
        home_xg_form, away_xg_form         rolling avg xG scored (last 5 games)
        home_xg_overperf, away_xg_overperf rolling avg (goals − xG) (last 5 games)
    """
    df = df.sort_values("date").reset_index(drop=True)

    # deque of (xg_scored, goals_scored) per team
    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    h_xgf, a_xgf, h_ovp, a_ovp = [], [], [], []

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        hh, ah = hist[home], hist[away]

        if hh:
            h_xgf.append(sum(xg for xg, _ in hh) / len(hh))
            h_ovp.append(sum(g - xg for xg, g in hh) / len(hh))
        else:
            h_xgf.append(XG_NEUTRAL)
            h_ovp.append(0.0)

        if ah:
            a_xgf.append(sum(xg for xg, _ in ah) / len(ah))
            a_ovp.append(sum(g - xg for xg, g in ah) / len(ah))
        else:
            a_xgf.append(XG_NEUTRAL)
            a_ovp.append(0.0)

        # update AFTER recording
        hist[home].append((row.home_xg, float(row.home_score)))
        hist[away].append((row.away_xg, float(row.away_score)))

    df["home_xg_form"]     = h_xgf
    df["away_xg_form"]     = a_xgf
    df["home_xg_overperf"] = h_ovp
    df["away_xg_overperf"] = a_ovp
    return df


# ── Step 5: coverage report ───────────────────────────────────────────────

def print_coverage(xg: pd.DataFrame, n_direct: int, n_swap: int, total: int):
    print(f"\n{'='*54}")
    print("  xG PATCH COVERAGE")
    print(f"{'='*54}")
    print(f"  Real xG matches (direct match) : {n_direct:>5}")
    print(f"  Real xG matches (swapped H/A)  : {n_swap:>5}")
    print(f"  Remaining neutral fills (1.0)  : {total - n_direct - n_swap:>5}")
    print(f"  Total rows                     : {total:>5}")

    print(f"\n  By competition:")
    for comp, grp in xg.groupby("competition"):
        print(f"    {comp:<30} {len(grp):>3} matches")

    print(f"\n  Sample — 5 rows with real xG:")
    sample_cols = ["date", "home_team", "away_team", "home_xg", "away_xg", "competition"]
    print(
        xg[sample_cols]
        .sample(5, random_state=42)
        .sort_values("date")
        .to_string(index=False)
    )
    print(f"{'='*54}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {DATA_PATH} …")
    main_df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"  {len(main_df):,} rows loaded")

    print("\nStep 1-2: Fetching StatsBomb xG …")
    xg_df = fetch_statsbomb_xg()
    print(f"  Built xG table: {len(xg_df)} matches")

    print("\nStep 3: Merging onto main dataset …")
    main_df, n_direct, n_swap = merge_xg(main_df, xg_df)

    print("\nStep 4: Recomputing rolling xG features …")
    main_df = compute_xg_rolling(main_df)

    print_coverage(xg_df, n_direct, n_swap, len(main_df))

    print(f"\nStep 5: Saving → {DATA_PATH} …")
    main_df.to_csv(DATA_PATH, index=False)
    print(f"  Saved {len(main_df):,} rows × {len(main_df.columns)} columns")


if __name__ == "__main__":
    main()
