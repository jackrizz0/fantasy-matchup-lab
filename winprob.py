"""Live and pregame win probabilities from ESPN's public scoreboard.

For each game the headline number is, in order of preference:
  live   — ESPN's in-game win probability (updates every play) once a game has started
  market — the betting moneyline with the bookmaker's margin removed, before kickoff
  model  — ESPN's Matchup Predictor (FPI) when no odds are posted yet
"""
from __future__ import annotations

import pandas as pd
import requests

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
TEAM_FIX = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}  # ESPN abbreviation -> nflverse


def _abbr(team: dict) -> str:
    a = team.get("abbreviation", "")
    return TEAM_FIX.get(a, a)


def _american_to_prob(odds) -> float | None:
    try:
        o = float(str(odds).replace("+", ""))
    except (TypeError, ValueError):
        return None
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def scoreboard(season: int, week: int) -> list[dict]:
    r = requests.get(SCOREBOARD, params={"seasontype": 2, "week": week, "dates": season}, timeout=15)
    r.raise_for_status()
    games = []
    for e in r.json().get("events", []):
        c = e["competitions"][0]
        teams = {x["homeAway"]: x for x in c["competitors"]}
        ids = {x["team"]["id"]: _abbr(x["team"]) for x in c["competitors"]}
        status = c["status"]
        sit = c.get("situation") or {}
        market = None
        if c.get("odds"):
            ml = c["odds"][0].get("moneyline") or {}
            ph = _american_to_prob(((ml.get("home") or {}).get("close") or {}).get("odds"))
            pa = _american_to_prob(((ml.get("away") or {}).get("close") or {}).get("odds"))
            if ph and pa:
                market = ph / (ph + pa)  # remove the bookmaker's margin
        games.append({
            "espn_id": e["id"], "kickoff": pd.Timestamp(e["date"]).tz_convert("America/New_York"),
            "away": _abbr(teams["away"]["team"]), "home": _abbr(teams["home"]["team"]),
            "away_score": int(teams["away"].get("score") or 0), "home_score": int(teams["home"].get("score") or 0),
            "state": status["type"]["state"],  # pre / in / post
            "detail": status["type"].get("shortDetail", ""),
            "possession": ids.get(sit.get("possession")), "down_distance": sit.get("downDistanceText"),
            "red_zone": bool(sit.get("isRedZone")), "market_home": market,
        })
    return games


def summary(espn_id: str) -> dict:
    """Matchup Predictor (pregame) and the play-by-play win probability series (live/final)."""
    r = requests.get(SUMMARY, params={"event": espn_id}, timeout=15)
    r.raise_for_status()
    s = r.json()
    pred = s.get("predictor") or {}
    try:
        fpi_home = float(pred["homeTeam"]["gameProjection"]) / 100
    except (KeyError, TypeError, ValueError):
        fpi_home = None
    series = [p["homeWinPercentage"] for p in (s.get("winprobability") or []) if "homeWinPercentage" in p]
    return {"fpi_home": fpi_home, "series": series}


def combine(games: list[dict], summaries: dict) -> pd.DataFrame:
    rows = []
    for g in games:
        s = summaries.get(g["espn_id"], {})
        series = s.get("series") or []
        fpi = s.get("fpi_home")
        if g["state"] != "pre" and series:
            home, source = series[-1], "live" if g["state"] == "in" else "final"
        elif g["market_home"] is not None:
            home, source = g["market_home"], "market"
        elif fpi is not None:
            home, source = fpi, "model"
        else:
            home, source = None, None
        if g["state"] == "post":  # a finished game is decided
            home = 1.0 if g["home_score"] > g["away_score"] else 0.0 if g["home_score"] < g["away_score"] else 0.5
            source = "final"
        rows.append({**g, "home_win": home, "source": source, "fpi_home": fpi, "series": series})
    return pd.DataFrame(rows)


def team_win_prob(table: pd.DataFrame) -> dict:
    """team -> (win probability, source)."""
    out = {}
    for _, g in table.iterrows():
        if g.home_win is None or pd.isna(g.home_win):
            continue
        out[g.home] = (g.home_win, g.source)
        out[g.away] = (1 - g.home_win, g.source)
    return out
