import pandas as pd

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
