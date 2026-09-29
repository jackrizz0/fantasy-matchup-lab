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


def _release(tag: str, name: str, max_age_hours: float = MAX_AGE_HOURS) -> pd.DataFrame | None:
    path = _fetch(f"{RELEASES}/{tag}/{name}", DATA_DIR / name, max_age_hours)
    return pd.read_parquet(path) if path else None


def load_all(season: int, force: bool = False) -> dict:
    """Current + prior season data. Prior-season files rarely change, so cache them for a week."""
    if force:
        for f in DATA_DIR.glob("*"):
            f.unlink()
    prev = season - 1
    week_old = 24 * 7
    games_path = _fetch(GAMES_URL, DATA_DIR / "games.csv")
    teams_path = _fetch(TEAMS_URL, DATA_DIR / "teams.csv", week_old)
    return {
        "season": season,
        "pbp": _release("pbp", f"play_by_play_{season}.parquet"),
        "pbp_prev": _release("pbp", f"play_by_play_{prev}.parquet", week_old),
        "ftn": _release("ftn_charting", f"ftn_charting_{season}.parquet"),
        # Personnel / formation / coverage data: use current season if published, else prior.
        "part": _release("pbp_participation", f"pbp_participation_{season}.parquet"),
        "part_prev": _release("pbp_participation", f"pbp_participation_{prev}.parquet", week_old),
        "roster": _release("rosters", f"roster_{season}.parquet"),
        "roster_prev": _release("rosters", f"roster_{prev}.parquet", week_old),
        "games": pd.read_csv(games_path) if games_path else None,
        # Team names, colors and logo URLs (for display only).
        "teams": pd.read_csv(teams_path).set_index("team_abbr") if teams_path else None,
    }
