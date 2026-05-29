import pandas as pd
import requests

FOOTBALL_DATA_API = "https://api.football-data.org/v4"


def fetch_match_lineups(match_id: int, api_key: str) -> dict:
    headers = {"X-Auth-Token": api_key}
    response = requests.get(
        f"{FOOTBALL_DATA_API}/matches/{match_id}",
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("lineups", {})


def lineups_to_dataframe(lineups: dict) -> pd.DataFrame:
    rows = []
    for team, data in lineups.items():
        for player in data.get("startXI", []):
            rows.append({
                "team": team,
                "player": player.get("name"),
                "position": player.get("pos"),
                "shirt_number": player.get("number"),
            })
    return pd.DataFrame(rows)
