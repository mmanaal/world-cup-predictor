import pandas as pd

_POSITION_GROUPS: dict[str, list[str]] = {
    "attack": ["CF", "ST", "LW", "RW", "SS"],
    "midfield": ["CM", "CAM", "CDM", "LM", "RM", "AM"],
    "defence": ["CB", "LB", "RB", "LWB", "RWB"],
}


def squad_average_rating(lineup: pd.DataFrame, player_ratings: dict[str, float]) -> float:
    return lineup["player"].map(player_ratings).mean()


def position_group_strength(
    lineup: pd.DataFrame,
    player_ratings: dict[str, float],
    group: str,
) -> float:
    positions = _POSITION_GROUPS.get(group, [])
    subset = lineup[lineup["position"].isin(positions)]
    if subset.empty:
        return 0.0
    return subset["player"].map(player_ratings).mean()


def build_lineup_features(
    home_lineup: pd.DataFrame,
    away_lineup: pd.DataFrame,
    player_ratings: dict[str, float],
) -> dict[str, float]:
    return {
        "home_squad_rating": squad_average_rating(home_lineup, player_ratings),
        "away_squad_rating": squad_average_rating(away_lineup, player_ratings),
        "home_attack_strength": position_group_strength(home_lineup, player_ratings, "attack"),
        "away_attack_strength": position_group_strength(away_lineup, player_ratings, "attack"),
        "home_defence_strength": position_group_strength(home_lineup, player_ratings, "defence"),
        "away_defence_strength": position_group_strength(away_lineup, player_ratings, "defence"),
    }
