"""Checks that the Model Scorecard backtest can't see the future.

For each completed week W, every current-season play from Week W onward is scrambled in the source data (yards,
touchdowns, targets, EPA, FTN charting). If project() only uses plays from before W, its Week W projections must be
identical with and without the scrambling.

Run:  python tests/test_backtest_leakage.py   (downloads data on first run, like the app)
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model  # noqa: E402
from data import load_all  # noqa: E402


def scramble_from(data: dict, week: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = dict(data)
    pbp = data["pbp"].copy()
    fut = pbp.week >= week
    for col in ["yards_gained", "passing_yards", "rushing_yards", "receiving_yards", "air_yards", "epa"]:
        pbp.loc[fut, col] = rng.normal(40, 30, fut.sum())
    for col in ["pass_touchdown", "rush_touchdown", "touchdown", "complete_pass", "interception", "sack"]:
        pbp.loc[fut, col] = rng.integers(0, 2, fut.sum())
    pbp.loc[fut, "receiver_player_id"] = rng.permutation(pbp.loc[fut, "receiver_player_id"].to_numpy())
    d["pbp"] = pbp
    fut_games = set(pbp.loc[fut, "game_id"])
    ftn = data["ftn"].copy()
    f = ftn.nflverse_game_id.isin(fut_games)
    ftn.loc[f, "read_thrown"] = rng.permutation(ftn.loc[f, "read_thrown"].to_numpy())
    ftn.loc[f, "n_blitzers"] = rng.integers(0, 8, f.sum())
    d["ftn"] = ftn
    return d


def check(data: dict, ppr: float = 1.0) -> list[str]:
    results = []
    actual = pd.concat([model.fantasy_by_game(data["pbp"], ppr), model.dst_by_game(data["pbp"], data["games"])])
    for w in model.completed_weeks(data):
        clean = model.data_through(data, w)
        assert clean["pbp"].empty or clean["pbp"].week.max() < w, f"Week {w}: later plays got through"
        assert clean["ftn"].nflverse_game_id.isin(set(clean["pbp"].game_id)).all(), f"Week {w}: FTN rows leak"
        team = actual[actual.week == w].groupby("player_id").team.first()
        a, _ = model.project(clean, ppr, w, team_asof=team)
        b, _ = model.project(model.data_through(scramble_from(data, w), w), ppr, w, team_asof=team)
        pd.testing.assert_series_equal(a.proj.sort_index(), b.proj.sort_index())
        # Sanity check that the scrambling is strong enough to matter when future data IS used.
        c, _ = model.project(scramble_from(data, w), ppr, w + 1, team_asof=team)
        d, _ = model.project(data, ppr, w + 1, team_asof=team)
        assert not np.allclose(c.proj.sort_index(), d.proj.reindex(c.proj.sort_index().index)), "scramble had no effect"
        seen = f"{len(clean['pbp'])} plays from Weeks 1–{w - 1}" if w > 1 else "no current-season plays (prior season only)"
        results.append(f"Week {w}: OK — {seen}, projections unchanged "
                       f"by scrambling Week {w}+ ({len(a)} players)")
    return results


def test_backtest_has_no_future_leakage():
    today = dt.date.today()
    season = today.year if today.month >= 9 else today.year - 1
    assert check(load_all(season))


if __name__ == "__main__":
    today = dt.date.today()
    for line in check(load_all(today.year if today.month >= 9 else today.year - 1)):
        print(line)
    print("No future data leakage.")
