import os

import pandas as pd
import requests

FOOTBALL_DATA_API = "https://api.football-data.org/v4"


def fetch_world_cup_matches(api_key: str, competition_code: str = "WC") -> pd.DataFrame:
    headers = {"X-Auth-Token": api_key}
    response = requests.get(
        f"{FOOTBALL_DATA_API}/competitions/{competition_code}/matches",
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    matches = response.json()["matches"]
    return pd.json_normalize(matches)


def load_historical_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])
