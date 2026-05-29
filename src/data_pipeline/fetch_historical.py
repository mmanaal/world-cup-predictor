import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

FOOTBALL_DATA_API = "https://api.football-data.org/v4"

def load_kaggle_results(path: str = "data/raw/results.csv") -> pd.DataFrame:
    """Load the Kaggle historical results CSV."""
    return pd.read_csv(path, parse_dates=["date"])

def fetch_recent_wc_matches(api_key: str, competition_code: str = "WC") -> pd.DataFrame:
    """Fetch current World Cup fixtures from football-data.org."""
    headers = {"X-Auth-Token": api_key}
    response = requests.get(
        f"{FOOTBALL_DATA_API}/competitions/{competition_code}/matches",
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    matches = response.json()["matches"]
    return pd.json_normalize(matches)