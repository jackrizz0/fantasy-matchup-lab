"""Downloads and caches free NFL data from nflverse (https://github.com/nflverse)."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TEAMS_URL = "https://raw.githubusercontent.com/nflverse/nflverse-pbp/master/teams_colors_logos.csv"
MAX_AGE_HOURS = 12

# Only the columns the app uses — loading all ~370 play-by-play columns needs well over 1 GB of memory.
PBP_COLUMNS = [
    "air_yards", "complete_pass", "defteam", "down", "epa", "extra_point_attempt", "extra_point_result",
    "field_goal_attempt", "field_goal_result", "fumble_lost", "fumble_recovery_1_team", "fumbled_1_player_id",
    "game_id", "interception", "kick_distance", "kicker_player_id", "no_huddle", "pass_attempt", "pass_touchdown",
    "passer_player_id", "passing_yards", "play_id", "play_type", "posteam", "punt_blocked", "qb_dropback",
    "qb_scramble", "receiver_player_id", "receiving_yards", "rush_attempt", "rush_touchdown", "rusher_player_id",
    "rushing_yards", "sack", "safety", "season", "season_type", "shotgun", "success", "td_player_id", "td_team",
    "touchdown", "two_point_attempt", "weather", "week", "wp", "yardline_100", "yards_gained",
]
PART_COLUMNS = ["nflverse_game_id", "play_id", "offense_personnel", "defense_personnel", "offense_formation",
                "defense_man_zone_type", "defense_coverage_type", "was_pressure"]
FTN_COLUMNS = ["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers", "n_defense_box",
               "is_motion", "is_play_action", "read_thrown", "is_catchable_ball", "is_contested_ball", "is_drop",
               "is_screen_pass", "is_rpo", "qb_location", "n_offense_backfield"]
NFLVERSE_FILES = ("play_by_play_*", "pbp_participation_*", "ftn_charting_*", "roster_*", "games.csv", "teams.csv")


def _fetch(url: str, dest: Path, max_age_hours: float = MAX_AGE_HOURS) -> Path | None:
    if dest.exists() and (time.time() - dest.stat().st_mtime) < max_age_hours * 3600:
        return dest
    DATA_DIR.mkdir(exist_ok=True)
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    except requests.RequestException:
        if not dest.exists():
            return None  # not published yet (e.g. participation for the current season)
    return dest


def _release(tag: str, name: str, max_age_hours: float = MAX_AGE_HOURS,
             columns: list[str] | None = None) -> pd.DataFrame | None:
    path = _fetch(f"{RELEASES}/{tag}/{name}", DATA_DIR / name, max_age_hours)
    return pd.read_parquet(path, columns=columns) if path else None


def load_all(season: int, force: bool = False) -> dict:
    """Current + prior season data. Prior-season files rarely change, so cache them for a week."""
    if force:  # re-download NFL data; keep the slow-to-rebuild weather history
        for pattern in NFLVERSE_FILES:
            for f in DATA_DIR.glob(pattern):
                f.unlink()
    prev = season - 1
    week_old = 24 * 7
    games_path = _fetch(GAMES_URL, DATA_DIR / "games.csv")
    teams_path = _fetch(TEAMS_URL, DATA_DIR / "teams.csv", week_old)
    return {
        "season": season,
        "pbp": _release("pbp", f"play_by_play_{season}.parquet", columns=PBP_COLUMNS),
        "pbp_prev": _release("pbp", f"play_by_play_{prev}.parquet", week_old, PBP_COLUMNS),
        "ftn": _release("ftn_charting", f"ftn_charting_{season}.parquet", columns=FTN_COLUMNS),
        # Personnel / formation / coverage data: use current season if published, else prior.
        "part": _release("pbp_participation", f"pbp_participation_{season}.parquet", columns=PART_COLUMNS),
        "part_prev": _release("pbp_participation", f"pbp_participation_{prev}.parquet", week_old, PART_COLUMNS),
        "roster": _release("rosters", f"roster_{season}.parquet"),
        "roster_prev": _release("rosters", f"roster_{prev}.parquet", week_old),
        "games": pd.read_csv(games_path) if games_path else None,
        # Team names, colors and logo URLs (for display only).
        "teams": pd.read_csv(teams_path).set_index("team_abbr") if teams_path else None,
    }
