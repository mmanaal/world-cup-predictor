import numpy as np
import pandas as pd


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
