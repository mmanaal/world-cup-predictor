from .fetch_historical import fetch_world_cup_matches, load_historical_csv
from .fetch_lineups import fetch_match_lineups, lineups_to_dataframe
from .preprocess import drop_incomplete_matches, encode_outcome, normalize_team_names

__all__ = [
    "fetch_world_cup_matches",
    "load_historical_csv",
    "fetch_match_lineups",
    "lineups_to_dataframe",
    "drop_incomplete_matches",
    "encode_outcome",
    "normalize_team_names",
]
