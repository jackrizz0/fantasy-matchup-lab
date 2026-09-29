"""Game weather: kickoff forecasts, historical conditions, and how teams, positions and players perform in them.

Forecasts and historical precipitation come from Open-Meteo (free, no key). Kickoff temperature and wind for
past games come from nflverse's schedule data.
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd
import requests

from data import DATA_DIR

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORY_START = 2019          # seasons of weather history used for splits
HISTORY_CACHE = DATA_DIR / "weather_history.parquet"
INDOOR_ROOFS = {"dome", "closed"}
WIND_MPH, COLD_F, HOT_F, PRECIP_IN = 15, 40, 85, 0.05   # PRECIP_IN = total over the 3 game hours

INDOORS, WINDY, WET, COLD, HOT = "Indoors", "Windy (15+ mph)", "Rain/Snow", "Cold (≤40°F)", "Hot (≥85°F)"
ADVERSE = [WINDY, WET, COLD, HOT]
CONDITIONS = [INDOORS] + ADVERSE

# nflverse stadium_id -> (lat, lon)
STADIUMS = {
    "ATL97": (33.755, -84.401), "BAL00": (39.278, -76.623), "BOS00": (42.091, -71.264), "BUF00": (42.774, -78.787),
    "CAR00": (35.226, -80.853), "CHI98": (41.862, -87.617), "CIN00": (39.095, -84.516), "CLE00": (41.506, -81.700),
    "DAL00": (32.748, -97.093), "DEN00": (39.744, -105.020), "DET00": (42.340, -83.046), "GNB00": (44.501, -88.062),
    "HOU00": (29.685, -95.411), "IND00": (39.760, -86.164), "JAX00": (30.324, -81.637), "KAN00": (39.049, -94.484),
    "LAX01": (33.953, -118.339), "LAX97": (33.864, -118.261), "LAX99": (34.014, -118.288), "MIA00": (25.958, -80.239),
    "MIN01": (44.974, -93.258), "MIN98": (44.976, -93.225), "NAS00": (36.166, -86.771), "NOR00": (29.951, -90.081),
    "NYC01": (40.814, -74.074), "OAK00": (37.752, -122.201), "PHI00": (39.901, -75.168), "PHO00": (33.528, -112.263),
    "PIT00": (40.447, -80.016), "SDG00": (32.783, -117.120), "SEA00": (47.595, -122.332), "SFO01": (37.403, -121.970),
    "TAM00": (27.976, -82.503), "VEG00": (36.091, -115.184), "WAS00": (38.908, -76.864),
    # international venues
    "LON00": (51.556, -0.280), "LON01": (51.456, -0.341), "LON02": (51.604, -0.066), "GER00": (48.219, 11.625),
    "MUN01": (48.219, 11.625), "FRA00": (50.069, 8.645), "MAD01": (40.453, -3.688), "MEX00": (19.303, -99.150),
    "SAO00": (-23.545, -46.474), "RIO00": (-22.912, -43.230), "PAR00": (48.924, 2.360), "MEL00": (-37.820, 144.983),
}


def condition_tags(indoor: bool, temp=np.nan, wind=np.nan, precip=np.nan, snow=np.nan) -> list[str]:
    if indoor:
        return [INDOORS]
    tags = []
    if pd.notna(wind) and wind >= WIND_MPH:
        tags.append(WINDY)
    if (pd.notna(precip) and precip >= PRECIP_IN) or (pd.notna(snow) and snow > 0):
        tags.append(WET)
    if pd.notna(temp) and temp <= COLD_F:
        tags.append(COLD)
    if pd.notna(temp) and temp >= HOT_F:
        tags.append(HOT)
    return tags


def _kickoff(gameday: str, gametime) -> pd.Timestamp:
    """Kickoff in US Eastern time, floored to the hour (schedule times are Eastern)."""
    t = str(gametime) if isinstance(gametime, str) and ":" in gametime else "13:00"
    return pd.Timestamp(f"{gameday} {t}").floor("h")


def _hourly(url: str, sid: str, start: str, end: str, fields: str) -> pd.DataFrame:
    lat, lon = STADIUMS[sid]
    for attempt in range(3):
        r = requests.get(url, timeout=90, params={
            "latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "hourly": fields,
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
            "timezone": "America/New_York"})
        if r.status_code != 429 or attempt == 2:
            break
        time.sleep(61)  # per-minute limit hit; wait it out
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    return h.assign(time=pd.to_datetime(h.time)).set_index("time")


def _window(h: pd.DataFrame, kickoff: pd.Timestamp) -> pd.DataFrame:
    return h.loc[kickoff: kickoff + pd.Timedelta(hours=2)]


# ---------------------------------------------------------------- history
def history(games: pd.DataFrame, pbp_frames: list) -> pd.DataFrame:
    """One row per completed game since HISTORY_START with kickoff conditions and tags.
    Precipitation is fetched once per stadium from Open-Meteo's archive and cached incrementally."""
    g = games[(games.season >= HISTORY_START) & games.home_score.notna()].copy()
    g["indoor"] = g.roof.isin(INDOOR_ROOFS)
    g["kickoff"] = [_kickoff(d, t) for d, t in zip(g.gameday, g.gametime)]

    cache = pd.read_parquet(HISTORY_CACHE) if HISTORY_CACHE.exists() else pd.DataFrame(columns=["game_id", "precip", "snow"])
    archive_end = pd.Timestamp(dt.date.today() - dt.timedelta(days=10))  # archive lags; newer games use pbp text
    need = g[~g.indoor & ~g.game_id.isin(cache.game_id) & g.stadium_id.isin(STADIUMS) & (g.kickoff < archive_end)]
    # One request per stadium-season (Sept-Feb only), sequentially: Open-Meteo rate-limits by data volume.
    # Progress is saved after every request, so an interrupted first run resumes where it left off.
    for (sid, _), grp in need.groupby(["stadium_id", "season"]):
        try:
            h = _hourly(ARCHIVE_URL, sid, grp.kickoff.min().strftime("%Y-%m-%d"),
                        grp.kickoff.max().strftime("%Y-%m-%d"), "precipitation,snowfall")
        except requests.RequestException:
            continue
        rows = [{"game_id": gid, "precip": w.precipitation.sum(), "snow": w.snowfall.sum()}
                for gid, k in zip(grp.game_id, grp.kickoff) if len(w := _window(h, k))]
        if rows:
            cache = pd.DataFrame(rows) if cache.empty else pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
            DATA_DIR.mkdir(exist_ok=True)
            cache.to_parquet(HISTORY_CACHE)
    g = g.merge(cache, on="game_id", how="left")

    # Recent games the archive doesn't cover yet: fall back to the play-by-play weather description.
    texts = pd.concat([p[["game_id", "weather"]].drop_duplicates("game_id") for p in pbp_frames if p is not None])
    wet_text = texts.set_index("game_id").weather.str.contains("rain|snow|shower|sleet", case=False, na=False)
    fallback = g.game_id.map(wet_text)
    g.loc[g.precip.isna() & (fallback == True), "precip"] = PRECIP_IN  # noqa: E712
    g["tags"] = [condition_tags(i, t, w, p, s) for i, t, w, p, s in zip(g.indoor, g.temp, g.wind, g.precip, g.snow)]
    return g


def team_splits(hist: pd.DataFrame) -> pd.DataFrame:
    """Per team and condition: games, points scored/allowed per game, and difference vs the team's overall."""
    long = pd.concat([
        pd.DataFrame({"team": hist.home_team, "pf": hist.home_score, "pa": hist.away_score, "tags": hist.tags}),
        pd.DataFrame({"team": hist.away_team, "pf": hist.away_score, "pa": hist.home_score, "tags": hist.tags}),
    ])
    overall = long.groupby("team")[["pf", "pa"]].mean()
    rows = []
    for cond in CONDITIONS:
        sub = long[long.tags.map(lambda t, c=cond: c in t)]
        for team, d in sub.groupby("team"):
            rows.append({"team": team, "condition": cond, "games": len(d), "ppg": d.pf.mean(), "allowed": d.pa.mean(),
                         "ppg_diff": d.pf.mean() - overall.at[team, "pf"],
                         "allowed_diff": d.pa.mean() - overall.at[team, "pa"]})
    return pd.DataFrame(rows).set_index(["team", "condition"])


def league_effects(hist: pd.DataFrame) -> pd.DataFrame:
    base = hist[hist.tags.map(len) == 0]
    base_total = (base.home_score + base.away_score).mean()
    rows = [{"condition": "Normal outdoor", "games": len(base), "avg_total_pts": base_total, "vs_normal": 0.0}]
    for cond in CONDITIONS:
        d = hist[hist.tags.map(lambda t, c=cond: c in t)]
        tot = (d.home_score + d.away_score).mean()
        rows.append({"condition": cond, "games": len(d), "avg_total_pts": tot, "vs_normal": tot - base_total})
    return pd.DataFrame(rows).set_index("condition")


def _tagged_fp(fp_all: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    return fp_all.merge(hist[["game_id", "tags"]], on="game_id", how="inner")


def position_effects(fp_all: pd.DataFrame, info: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Fantasy multiplier per position and condition: points in that condition vs the same players' averages,
    shrunk toward 1.0 for small samples."""
    fp = _tagged_fp(fp_all, hist).join(info.position, on="player_id")
    avg = fp.groupby("player_id").pts.transform("mean")
    fp = fp[avg >= 3].assign(player_avg=avg[avg >= 3])  # ignore fringe players whose averages are ~0
    rows = []
    for pos, d in fp.groupby("position"):
        for cond in ADVERSE:
            sub = d[d.tags.map(lambda t, c=cond: c in t)]
            n = sub.game_id.nunique()
            ratio = sub.pts.sum() / sub.player_avg.sum() if len(sub) else 1.0
            mult = float(np.clip(1 + (ratio - 1) * n / (n + 40), 0.85, 1.10))
            rows.append({"position": pos, "condition": cond, "games": n, "raw_ratio": ratio, "mult": mult})
    return pd.DataFrame(rows).set_index(["position", "condition"])


def player_splits(fp_all: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    fp = _tagged_fp(fp_all, hist)
    rows = []
    for cond in CONDITIONS:
        hit = fp.tags.map(lambda t, c=cond: c in t)
        a = fp[hit].groupby("player_id").pts.agg(["size", "mean"])
        b = fp[~hit].groupby("player_id").pts.mean()
        rows.append(pd.DataFrame({"condition": cond, "games": a["size"], "ppg": a["mean"], "ppg_other": b.reindex(a.index)}))
    return pd.concat(rows).reset_index().set_index(["player_id", "condition"])


# ---------------------------------------------------------------- forecast
def forecast(week_games: pd.DataFrame) -> pd.DataFrame:
    """Kickoff-window forecast per game (Open-Meteo gives ~16 days ahead). One request covers every stadium."""
    today = dt.date.today()
    horizon = pd.Timestamp(today + dt.timedelta(days=15))
    rows, todo = [], []
    for _, r in week_games.iterrows():
        k = _kickoff(r.gameday, r.gametime)
        base = {"game_id": r.game_id, "home_team": r.home_team, "away_team": r.away_team, "kickoff": k,
                "stadium": r.stadium, "roof": r.roof, "tags": []}
        if r.roof in INDOOR_ROOFS:
            rows.append({**base, "tags": [INDOORS], "summary": f"Indoors ({r.roof} roof)"})
        elif r.stadium_id not in STADIUMS or k > horizon or k.date() < today:
            rows.append({**base, "summary": "Forecast not available yet"})
        else:
            todo.append((base, STADIUMS[r.stadium_id]))
    if todo:
        fields = "temperature_2m,precipitation_probability,precipitation,snowfall,wind_speed_10m,wind_gusts_10m"
        try:
            r = requests.get(FORECAST_URL, timeout=30, params={
                "latitude": ",".join(str(lat) for _, (lat, _) in todo),
                "longitude": ",".join(str(lon) for _, (_, lon) in todo),
                "start_date": min(b["kickoff"] for b, _ in todo).strftime("%Y-%m-%d"),
                "end_date": max(b["kickoff"] for b, _ in todo).strftime("%Y-%m-%d"),
                "hourly": fields, "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "precipitation_unit": "inch", "timezone": "America/New_York"})
            r.raise_for_status()
            payload = r.json()
            payload = payload if isinstance(payload, list) else [payload]
        except (requests.RequestException, ValueError):
            payload = None
        for i, (base, _) in enumerate(todo):
            if payload is None:
                rows.append({**base, "summary": "Forecast unavailable (couldn't reach Open-Meteo)"})
                continue
            h = pd.DataFrame(payload[i]["hourly"])
            w = _window(h.assign(time=pd.to_datetime(h.time)).set_index("time"), base["kickoff"])
            temp, wind, gust = w.temperature_2m.iloc[0], w.wind_speed_10m.max(), w.wind_gusts_10m.max()
            precip, snow, pop = w.precipitation.sum(), w.snowfall.sum(), w.precipitation_probability.max()
            summary = f"{temp:.0f}°F, wind {wind:.0f} mph (gusts {gust:.0f}), {pop:.0f}% precip chance"
            if precip > 0:
                summary += f", {precip:.2f}\" expected" + (" (snow)" if snow > 0 else "")
            rows.append({**base, "temp": temp, "wind": wind, "gusts": gust, "precip_prob": pop, "precip": precip,
                         "snow": snow, "tags": condition_tags(False, temp, wind, precip, snow), "summary": summary})
    return pd.DataFrame(rows).set_index("game_id").sort_values("kickoff") if rows else pd.DataFrame()


def by_team(fc: pd.DataFrame) -> dict:
    """team -> forecast row, for both teams in each game."""
    out = {}
    for _, r in fc.iterrows():
        out[r.home_team] = r
        out[r.away_team] = r
    return out


def apply(proj: pd.DataFrame, fc: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    """Multiply projections by the position's historical effect for each adverse forecast condition."""
    teams = by_team(fc)
    proj = proj.copy()
    mults, tags = [], []
    for _, r in proj.iterrows():
        f = teams.get(r.team)
        t = list(f.tags) if f is not None else []
        m = 1.0
        for cond in t:
            if (r.position, cond) in effects.index:
                m *= effects.at[(r.position, cond), "mult"]
        mults.append(float(np.clip(m, 0.8, 1.12)))
        tags.append(", ".join(t))
    proj["weather_mult"], proj["weather"] = mults, tags
    proj["proj"] = proj.proj * proj.weather_mult
    return proj.sort_values("proj", ascending=False)


def notes(r: pd.Series, fc: pd.DataFrame, tsplits: pd.DataFrame, psplits: pd.DataFrame) -> list[str]:
    f = by_team(fc).get(r.team)
    if f is None:
        return []
    out = [f"Forecast: {f.summary}."]
    for cond in [c for c in f.tags if c in ADVERSE]:
        if (r.team, cond) in tsplits.index:
            s = tsplits.loc[(r.team, cond)]
            out.append(f"Since {HISTORY_START}, {r.team} scores {s.ppg:.1f} pts/gm in {cond.lower()} games "
                       f"({s.ppg_diff:+.1f} vs its norm, {s.games:.0f} games).")
        if (r.name, cond) in psplits.index:
            p = psplits.loc[(r.name, cond)]
            if p.games >= 2 and pd.notna(p.ppg_other):
                out.append(f"{r['name']} has averaged {p.ppg:.1f} fantasy pts in {cond.lower()} games "
                           f"vs {p.ppg_other:.1f} otherwise ({p.games:.0f} games, last 2 seasons).")
    return out
