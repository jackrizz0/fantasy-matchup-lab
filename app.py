"""Fantasy Matchup Lab — run with:  streamlit run app.py"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

import data as nfl_data
import importers
import model
import ui
import weather
import winprob

st.set_page_config(page_title="Fantasy Matchup Lab", page_icon="🏈", layout="wide")

# The app's own modules, in reload order (each after the modules it imports).
LOCAL_MODULES = [nfl_data, weather, winprob, model, importers, ui]


def _code_version() -> str:
    h = hashlib.sha256()
    for m in LOCAL_MODULES:
        h.update(Path(m.__file__).read_bytes())
    return h.hexdigest()


@st.cache_resource
def _loaded_code() -> dict:
    return {"version": _code_version()}


def reload_if_code_changed():
    """Streamlit Community Cloud pulls pushed code without watching files: app.py re-runs with the new code, but
    the imported modules (model, data, ...) and the cached results stay as they were at startup until a reboot.
    So when any module's source changes, reload the modules and drop every cache (the NFL data re-downloads)."""
    version = _code_version()
    if _loaded_code()["version"] == version:
        return
    for m in LOCAL_MODULES:
        importlib.reload(m)
    st.cache_data.clear()
    st.cache_resource.clear()
    _loaded_code()["version"] = version


reload_if_code_changed()

SCORING = {"PPR": 1.0, "Half PPR": 0.5, "Standard": 0.0}
DEFAULT_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "SUPERFLEX": 0, "K": 1, "DST": 1}
SLOT_MAX = {"QB": 2, "RB": 4, "WR": 5, "TE": 3, "FLEX": 4, "SUPERFLEX": 2, "K": 2, "DST": 2}
SLOT_LABELS = {"DST": "D/ST"}


@st.cache_data(ttl=300, show_spinner="Fetching your league…")
def cached(fn_name: str, *args):
    """Cache fantasy-site API calls for 5 minutes so reruns don't re-hit their servers."""
    return getattr(importers, fn_name)(*args)


def apply_import(team: dict):
    """Button callback: runs before the rerun, so it may set keyed widget values."""
    for slot in DEFAULT_SLOTS:
        st.session_state[f"slot_{slot}"] = min(team["slots"].get(slot, 0), SLOT_MAX[slot])
    st.session_state["scoring"] = {v: k for k, v in SCORING.items()}[team["ppr"]]
    st.session_state["import_ids"] = team["player_ids"]
    st.session_state["import_msg"] = (team["team"], team["unmatched"])


def yahoo_config() -> dict:
    """Site-wide Yahoo app, from .streamlit/secrets.toml (or the host's secrets settings):
        [yahoo]
        client_id = "..."
        client_secret = "..."
        redirect_uri = "https://your-app.streamlit.app"   # omit when running locally (uses "oob")
    """
    try:
        return dict(st.secrets.get("yahoo", {}))
    except Exception:  # no secrets file at all
        return {}


def _yahoo_creds() -> tuple[str, str, str]:
    cfg = yahoo_config()
    cid, secret = st.session_state.get("yahoo_creds", (cfg.get("client_id", ""), cfg.get("client_secret", "")))
    return cid, secret, cfg.get("redirect_uri", "oob")


def yahoo_handle_redirect():
    """Back from Yahoo's sign-in page (hosted setup): the URL carries ?code=...&state=yahoo."""
    qp = st.query_params
    if qp.get("state") != "yahoo" or not qp.get("code"):
        return
    code = qp["code"]
    del qp["code"], qp["state"]
    cid, secret, redirect = _yahoo_creds()
    st.session_state["import_platform"] = "Yahoo"
    try:
        st.session_state["yahoo_token"] = importers.yahoo_exchange_code(cid, secret, code, redirect)
    except importers.LeagueImportError as e:
        st.session_state["yahoo_error"] = str(e)


def _session_memo(key, fn):
    """Per-visitor 5-minute memo for signed-in calls (Yahoo, private ESPN leagues). The shared cache would keep one
    visitor's credentials and leagues in memory where another visitor's identical request could reuse them."""
    memo = st.session_state.setdefault("session_memo", {})
    hit = memo.get(key)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    value = fn()
    memo[key] = (time.time(), value)
    return value


def yahoo_section(season: int, maps: dict):
    """Each visitor signs in to their own Yahoo account; the token lives only in their browser session."""
    if "yahoo_error" in st.session_state:
        st.error(st.session_state.pop("yahoo_error"))
    cid, secret, redirect = _yahoo_creds()
    tok = st.session_state.get("yahoo_token")
    if tok is None:
        if cid and secret and redirect != "oob":  # hosted: one click, Yahoo sends the visitor back here
            st.link_button("Sign in with Yahoo", importers.yahoo_auth_url(cid, redirect, "yahoo"), type="primary")
            st.caption("You'll approve read-only access to your Yahoo Fantasy leagues. The sign-in lasts for this "
                       "browser session only and isn't stored.")
            return None
        if not (cid and secret):
            st.markdown(
                "**One-time setup** (about 2 minutes):\n"
                "1. Go to [developer.yahoo.com/apps/create](https://developer.yahoo.com/apps/create). "
                "Name it anything, set **Redirect URI** to `oob`, and tick **Fantasy Sports → Read**.\n"
                "2. Paste the **Client ID** and **Client Secret** Yahoo gives you below.\n"
                "3. Click the authorize link, approve, and paste the code Yahoo shows you.")
            c1, c2 = st.columns(2)
            cid = c1.text_input("Client ID", type="password", key="y_cid")
            secret = c2.text_input("Client Secret", type="password", key="y_secret")
        if cid:
            st.markdown(f"[👉 Authorize with Yahoo]({importers.yahoo_auth_url(cid)})")
        code = st.text_input("Code from Yahoo", key="y_code")
        if st.button("Connect Yahoo", disabled=not (cid and secret and code)):
            st.session_state["yahoo_token"] = importers.yahoo_exchange_code(cid, secret, code)
            st.session_state["yahoo_creds"] = (cid, secret)
            st.rerun()
        st.caption("Your Yahoo sign-in is kept only in this browser session — it's gone when you refresh or close "
                   "the page — and is only sent to Yahoo. To skip retyping the Client ID and Secret, put them in "
                   "`.streamlit/secrets.toml` under `[yahoo]`.")
        return None
    tok = importers.yahoo_fresh(tok, cid, secret, redirect)
    st.session_state["yahoo_token"] = tok
    c1, c2 = st.columns([3, 1])
    c1.success("Yahoo connected (this browser session only).")
    if c2.button("Disconnect"):
        for k in ("yahoo_token", "yahoo_creds", "session_memo"):
            st.session_state.pop(k, None)
        st.rerun()
    leagues = _session_memo(("leagues", season), lambda: importers.yahoo_leagues(tok["access_token"], season))
    if not leagues:
        st.warning(f"No {season} Yahoo leagues found on this account.")
        return None
    league = st.selectbox("League", leagues, format_func=lambda lg: lg["name"])
    return _session_memo(("teams", league["league_key"]),
                       lambda: importers.yahoo_teams(tok["access_token"], league["league_key"], maps))


def import_panel(season: int, data: dict):
    yahoo_handle_redirect()
    with st.expander("📥 Import your team from Sleeper, ESPN or Yahoo",
                     expanded=not st.query_params.get_all("p") or "yahoo_token" in st.session_state):
        platform = st.radio("Platform", ["Sleeper", "ESPN", "Yahoo"], horizontal=True, key="import_platform")
        maps = get_id_maps(season)
        teams = None
        try:
            if platform == "Sleeper":
                c1, c2 = st.columns(2)
                username = c1.text_input("Sleeper username", key="sleeper_user")
                league_id = c2.text_input("…or league ID", key="sleeper_league")
                user_id = cached("sleeper_user_id", username) if username else None
                if user_id and not league_id:
                    leagues = cached("sleeper_leagues", user_id, season)
                    if not leagues:
                        st.warning(f"No {season} leagues found for {username}.")
                    else:
                        league_id = st.selectbox("League", leagues, format_func=lambda lg: lg["name"])["league_id"]
                if league_id:
                    teams = cached("sleeper_teams", league_id, maps, user_id)
            elif platform == "Yahoo":
                teams = yahoo_section(season, maps)
            else:
                league_id = st.text_input("ESPN league ID (the leagueId=… number in your league's URL)", key="espn_league")
                st.caption("Private league? Add two cookies from your browser while logged in to ESPN "
                           "(Developer Tools → Application → Cookies → espn.com). They stay on this computer "
                           "and are only sent to ESPN.")
                c1, c2 = st.columns(2)
                s2 = c1.text_input("espn_s2 (private leagues only)", type="password", key="espn_s2")
                swid = c2.text_input("SWID (private leagues only)", type="password", key="espn_swid")
                if league_id:
                    if s2 and swid:  # private league: keep the visitor's cookies out of the shared cache
                        teams = _session_memo(("espn", league_id, season),
                                              lambda: importers.espn_teams(league_id, season, maps, s2, swid))
                    else:
                        teams = cached("espn_teams", league_id, season, maps, "", "")
        except importers.LeagueImportError as e:
            st.error(str(e))
        except Exception as e:  # unexpected API shape
            st.error(f"Import failed: {e}")
        if teams:
            teams = sorted(teams, key=lambda t: not t["mine"])  # your team first when we can tell
            team = st.selectbox("Your team", teams,
                                format_func=lambda t: f"{t['team']} ({t['owner']})" + (" ← you" if t["mine"] else ""))
            st.caption(f"{len(team['player_ids'])} players · {team['ppr']:g} PPR · slots: "
                       + ", ".join(f"{n} {SLOT_LABELS.get(s, s)}" for s, n in team["slots"].items()))
            st.button("Import this roster & league settings", type="primary", on_click=apply_import, args=(team,))
    if "import_msg" in st.session_state:
        name, unmatched = st.session_state.pop("import_msg")
        st.success(f"Imported {name}.")
        if unmatched:
            st.warning("Couldn't match: " + ", ".join(unmatched)
                       + ". They may be injured/inactive or missing from NFL roster data — add them manually if needed.")


# cache_resource shares one read-only copy between all visitors; cache_data would hand every caller a fresh copy
# of the play-by-play tables, multiplying memory use. Nothing downstream modifies these frames in place.
@st.cache_resource(ttl=6 * 3600, show_spinner="Downloading NFL data from nflverse…")
def get_data(season: int, force: bool = False):
    return nfl_data.load_all(season, force)


@st.cache_data(ttl=6 * 3600)
def get_id_maps(season: int):
    return importers.id_maps(get_data(season))


@st.cache_data(ttl=6 * 3600, show_spinner="Crunching matchups…")
def get_projections(season: int, ppr: float, week: int):
    return model.project(get_data(season), ppr, week)


@st.cache_data(ttl=6 * 3600, show_spinner="Finding usage shifts…")
def get_usage_shifts(season: int, ppr: float, week: int):
    return model.usage_shifts(get_projections(season, ppr, week)[1])


@st.cache_data(ttl=6 * 3600, show_spinner="Backtesting every completed week…")
def get_backtest(season: int, ppr: float, through: int):
    """`through` (latest week with stats) is only part of the cache key, so a new week triggers a fresh run."""
    return model.backtest(get_data(season), ppr)


@st.cache_data(ttl=6 * 3600, show_spinner="Loading weather history (first run takes a minute or two)…")
def get_weather_history(season: int):
    d = get_data(season)
    return weather.history(d["games"], [d["pbp"], d["pbp_prev"]])


@st.cache_data(ttl=3600, show_spinner="Fetching kickoff forecasts…")
def get_forecast(season: int, week: int):
    g = get_data(season)["games"]
    return weather.forecast(g[(g.season == season) & (g.week == week) & (g.game_type == "REG")])


@st.cache_data(ttl=6 * 3600)
def get_weather_model(season: int, ppr: float, week: int):
    _, c = get_projections(season, ppr, week)
    hist = get_weather_history(season)
    return (weather.position_effects(c["fp_all"], c["info"], hist), weather.team_splits(hist),
            weather.player_splits(c["fp_all"], hist), weather.league_effects(hist))


# Win probabilities: the scoreboard is cheap and refreshed often; each game's detail is cached by its state
# (pregame odds/predictor for an hour, live win probability for ~30 seconds, finished games for a day).
@st.cache_data(ttl=25, show_spinner=False)
def get_scoreboard(season: int, week: int):
    try:
        return winprob.scoreboard(season, week)
    except (requests.RequestException, ValueError, KeyError):
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def summary_pre(espn_id: str):
    return winprob.summary(espn_id)


@st.cache_data(ttl=25, show_spinner=False)
def summary_live(espn_id: str):
    return winprob.summary(espn_id)


@st.cache_data(ttl=86400, show_spinner=False)
def summary_final(espn_id: str):
    return winprob.summary(espn_id)


def still_live(season: int, week: int) -> pd.DataFrame:
    """Games from an earlier week that are still being played (e.g. Monday night while the app shows next week)."""
    if week < 1:
        return pd.DataFrame()
    games = [g for g in get_scoreboard(season, week) if g["state"] == "in"]
    summaries = {}
    for g in games:
        try:
            summaries[g["espn_id"]] = summary_live(g["espn_id"])
        except (requests.RequestException, ValueError, KeyError):
            pass
    return winprob.combine(games, summaries) if games else pd.DataFrame()


def get_winprobs(season: int, week: int) -> pd.DataFrame:
    games = get_scoreboard(season, week)
    fetch = {"pre": summary_pre, "in": summary_live, "post": summary_final}
    summaries = {}
    for g in games:
        try:
            summaries[g["espn_id"]] = fetch.get(g["state"], summary_pre)(g["espn_id"])
        except (requests.RequestException, ValueError, KeyError):
            pass
    return winprob.combine(games, summaries)



# ---------------------------------------------------------------- sidebar
ui.inject_css()
today = dt.date.today()
default_season = today.year if today.month >= 9 else today.year - 1
with st.sidebar:
    st.markdown("## 🏈 Matchup Lab")
    c1, c2 = st.columns(2)
    season = c1.number_input("Season", 2016, today.year, default_season)
    data = get_data(season)
    if data["pbp"] is None:
        st.error(f"No {season} play-by-play published yet.")
        st.stop()
    week = c2.number_input("Week", 1, 18, model.default_week(data["games"], season))
    # Keyed widgets so an imported league can set scoring and lineup slots.
    st.session_state.setdefault("scoring", "PPR")
    for slot, n in DEFAULT_SLOTS.items():
        st.session_state.setdefault(f"slot_{slot}", n)
    scoring = st.segmented_control("Scoring", list(SCORING), key="scoring") or "PPR"
    ppr = SCORING[scoring]
    with st.expander("Lineup slots", expanded=False):
        c1, c2 = st.columns(2)
        slots = {slot: (c1, c2)[i % 2].number_input(SLOT_LABELS.get(slot, slot), 0, SLOT_MAX[slot], key=f"slot_{slot}")
                 for i, slot in enumerate(DEFAULT_SLOTS)}
    st.caption("Imported leagues set scoring and slots automatically.")
    st.divider()
    if st.button("↻ Refresh data", width="stretch"):
        st.cache_data.clear()
        get_data.clear()
        get_data(season, force=True)
        st.rerun()
    st.caption("Stats refresh every few hours; forecasts hourly.")

proj, ctx = get_projections(season, ppr, week)
effects, tsplits, psplits, league_wx = get_weather_model(season, ppr, week)
fc = get_forecast(season, week)
proj = weather.apply(proj, fc, effects)
teams = ui.Teams(data.get("teams"))
fc_by_team = weather.by_team(fc)
wp = get_winprobs(season, week)
team_wp = winprob.team_win_prob(wp) if len(wp) else {}
proj["win_prob"] = proj.team.map(lambda t: team_wp.get(t, (None, None))[0]).astype(float)
proj["win_source"] = proj.team.map(lambda t: team_wp.get(t, (None, None))[1])
defense, offense, sched = ctx["defense"], ctx["offense"], ctx["schedule"]
pers_season = defense.attrs.get("personnel_season")
ftn_through = ctx.get("ftn_through")
label = lambda pid: f"{proj.at[pid, 'name']} ({proj.at[pid, 'position']}, {proj.at[pid, 'team']})"


def notes(r: pd.Series) -> list[str]:
    return model.matchup_notes(r, ctx) + weather.notes(r, fc, tsplits, psplits)


# ---------------------------------------------------------------- header
played_through = int(ctx["fp_cur"].week.max()) if len(ctx["fp_cur"]) else 0
wx_games = int(fc.tags.map(lambda t: any(c in weather.ADVERSE for c in t)).sum()) if len(fc) else 0
prev_live = still_live(season, week - 1)
live_now = int((wp.state == "in").sum()) + len(prev_live) if len(wp) else len(prev_live)
ui.hero(f"{season} season · {scoring}", "MATCHUP LAB",
        "Opponent scheme, personnel, weather and live win odds — turned into your best lineup.",
        str(week), "week",
        [f"🔴 {live_now} live now" if live_now else f"🏈 {len(sched) // 2} games this week",
         f"🌦️ {wx_games} weather games" if wx_games else "☀️ No weather concerns",
         f"📊 Stats through Week {played_through}", f"🧩 Personnel data: {pers_season}"]
        + ([f"📝 Charting: {season} thru Wk {ftn_through}"] if ftn_through else []))


@st.fragment(run_every=30 if live_now else None)
def live_scoreboard():
    """Re-runs on its own every 30s while games are live, without reloading the rest of the page."""
    st.markdown(ui.scoreboard_strip(pd.concat([still_live(season, week - 1), get_winprobs(season, week)],
                                              ignore_index=True), teams), unsafe_allow_html=True)


live_scoreboard()

DISPLAY = {"name": "Player", "position": "Pos", "team": "Team", "opp": "Opp", "proj": "Proj",
           "ppg": f"{season} PPG", "last3": "Last 3", "ppg_prev": f"{season - 1} PPG",
           "tgt_share": "Tgt %", "carry_share": "Carry %", "adot": "aDOT", "rz": "RZ opps",
           "first_read_share": "1st-read %", "catchable_rate": "Catchable %", "contested_rate": "Contested %",
           "drops": "Drops",
           "dvp_mult": "Matchup ×", "scheme_mult": "Scheme ×", "env_mult": "Vegas ×", "weather_mult": "Weather ×",
           "implied_total": "Implied pts", "win_prob": "Win %", "weather": "Weather"}
PERSONNEL_COLS = {
    "plays": st.column_config.NumberColumn("Plays", format="%d"),
    "share": st.column_config.ProgressColumn("Usage", format="percent", min_value=0, max_value=1),
    "pass_rate": st.column_config.NumberColumn("Pass rate", format="percent"),
    "success_rate": st.column_config.NumberColumn("Success", format="percent"),
    "epa_per_play": st.column_config.NumberColumn("EPA / play", format="%+.2f"),
}
MULTS = ["Matchup ×", "Scheme ×", "Vegas ×", "Weather ×"]
TQ_HELP = {
    "1st-read %": f"Share of the team's first-read throws that went to this player, in games they played "
                  f"({season} FTN charting).",
    "Catchable %": f"Share of this player's targets that were catchable balls ({season} FTN charting).",
    "Contested %": f"Share of this player's targets thrown into tight coverage ({season} FTN charting).",
    "Drops": f"Charted drops this season ({season} FTN charting).",
}


def photo(r) -> str:
    return teams.logo(r.team) if r.position == "DST" or not isinstance(r.get("headshot"), str) else r.headshot


def table(df: pd.DataFrame, height: int | str = "auto"):
    view = df.assign(photo=[photo(r) for _, r in df.iterrows()])
    out = view[["photo"] + [c for c in DISPLAY if c in view]].rename(columns=DISPLAY)
    mults = [m for m in MULTS if m in out]
    rates = [c for c in ["1st-read %", "Catchable %", "Contested %"] if c in out]
    styled = out.style.map(ui.mult_color, subset=mults).format(
        {**{m: "{:.2f}" for m in mults}, **{c: "{:.0%}" for c in rates},
         **({"Drops": "{:.0f}"} if "Drops" in out else {})}, na_rep="–")
    pct = {c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
           for c in ["Tgt %", "Carry %", "Win %"]}
    nums = {c: st.column_config.NumberColumn(c, format="%.1f")
            for c in [f"{season} PPG", "Last 3", f"{season - 1} PPG", "aDOT", "Implied pts"]}
    nums |= {c: st.column_config.NumberColumn(c, help=h) for c, h in TQ_HELP.items()}
    st.dataframe(styled, hide_index=True, width="stretch", height=height, column_config={
        "photo": st.column_config.ImageColumn("", width="small"),
        "Player": st.column_config.TextColumn("Player", width="medium"),
        "Proj": st.column_config.ProgressColumn("Proj", format="%.1f", min_value=0,
                                                max_value=float(max(proj.proj.max(), 1))),
        "RZ opps": st.column_config.NumberColumn("RZ opps", format="%d"),
        **pct, **nums})


def win_panel(g: pd.Series, away: str, home: str):
    """Win-probability numbers from each source, plus the live chart once the game has started."""
    c1, c2, c3 = st.columns(3)
    mh, fh = g.market_home, g.fpi_home
    c1.metric("Betting market", f"{home} {mh:.0%}" if mh is not None and pd.notna(mh) else "–",
              f"{away} {1 - mh:.0%}" if mh is not None and pd.notna(mh) else None, delta_color="off",
              help="Moneyline odds with the bookmaker's margin removed.")
    c2.metric("ESPN predictor", f"{home} {fh:.0%}" if fh is not None and pd.notna(fh) else "–",
              f"{away} {1 - fh:.0%}" if fh is not None and pd.notna(fh) else None, delta_color="off",
              help="ESPN's pregame Matchup Predictor (FPI).")
    live = g.series[-1] if g.series else None
    c3.metric("Live win probability" if g.state == "in" else "In-game win prob.",
              f"{home} {ui.pct(live, g.state == 'in')}" if live is not None else "Not started",
              f"{away} {ui.pct(1 - live, g.state == 'in')}" if live is not None else None, delta_color="off",
              help="ESPN's in-game model, updated after every play.")
    if g.series:
        import altair as alt
        d = pd.DataFrame({"play": range(len(g.series)), "home": g.series})
        base = alt.Chart(d).encode(x=alt.X("play:Q", title=None, axis=alt.Axis(labels=False, ticks=False)))
        area = base.mark_area(line={"color": teams.color(home), "strokeWidth": 2}, opacity=.35,
                              color=teams.color(home)).encode(
            y=alt.Y("home:Q", title=f"{home} win %", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("home:Q", title=f"{home} win %", format=".0%")])
        mid = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="#6B7894").encode(y="y:Q")
        st.altair_chart((area + mid).properties(height=200), width="stretch")


def card_grid(rows: pd.DataFrame, slot_of, ncols: int = 3):
    for start in range(0, len(rows), ncols):
        cols = st.columns(ncols)
        for col, (pid, r) in zip(cols, list(rows.iterrows())[start:start + ncols]):
            with col:
                st.markdown(ui.player_card(r, slot_of(r), teams, fc_by_team.get(r.team), defense),
                            unsafe_allow_html=True)
                with st.expander("Why this projection"):
                    for n in notes(r):
                        st.markdown(f'<div class="ml-note">• {n}</div>', unsafe_allow_html=True)


def metric_grid(items: list, ncols: int = 3):
    """items: (label, value, league_avg, kind) where kind is 'pct', 'num' or 'epa'."""
    fmt = {"pct": lambda v: f"{v:.0%}", "num": lambda v: f"{v:.1f}", "epa": lambda v: f"{v:+.3f}"}
    dfmt = {"pct": lambda d: f"{d * 100:+.0f} pts vs lg", "num": lambda d: f"{d:+.1f} vs lg",
            "epa": lambda d: f"{d:+.3f} vs lg"}
    cols = st.columns(ncols)
    for i, (lab, v, lg, kind) in enumerate(items):
        if v is None or pd.isna(v):
            cols[i % ncols].metric(lab, "–")
        else:
            cols[i % ncols].metric(lab, fmt[kind](v), dfmt[kind](v - lg) if pd.notna(lg) else None, delta_color="off")


tab_lineup, tab_rank, tab_shift, tab_game, tab_wx, tab_def, tab_score = st.tabs(
    ["🧑‍🤝‍🧑 My Lineup", "📋 Rankings", "📈 Usage Shifts", "🔎 Game Breakdown", "🌦️ Weather", "🛡️ Defenses",
     "🎯 Model Scorecard"])

# ---------------------------------------------------------------- my lineup
with tab_lineup:
    import_panel(season, data)
    if "import_ids" in st.session_state:
        imported = st.session_state.pop("import_ids")
        st.query_params["p"] = imported
        missing = [p for p in imported if p not in proj.index]
        if missing:
            names = ctx["info"].name.reindex(missing).fillna("unknown player")
            st.info("On your roster but without a projection (no recent games): " + ", ".join(names))
    saved = [p for p in st.query_params.get_all("p") if p in proj.index]
    roster = st.multiselect("Your roster", proj.index.tolist(), default=saved, format_func=label,
                            placeholder="Search players, kickers or D/STs (e.g. “MIN D/ST”)…",
                            help="Your roster is saved in the page link — bookmark it to come back to it.")
    if roster != saved:
        st.query_params["p"] = roster
    if not roster:
        st.info("Import your team above or add players to get your optimal lineup, with the reasons behind every pick.")
    else:
        lineup = model.optimize_lineup(proj.loc[roster], slots)
        bench = proj.loc[[p for p in roster if p not in lineup.index]].sort_values("proj", ascending=False)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Projected points", f"{lineup.proj.sum():.1f}")
        m2.metric("Strong matchups", int((lineup.dvp_raw >= 1.04).sum()), help="Starters facing a defense that "
                  "allows more fantasy points than average to their position.")
        m3.metric("Weather-affected starters", int((lineup.weather_mult.sub(1).abs() > 0.005).sum()))
        byes = int(lineup.opp.isna().sum())
        m4.metric("Starters on bye", byes, help="Swap these out!" if byes else None)
        ui.section("Starting lineup")
        slot_by_id = dict(zip(lineup.index, lineup.slot))
        card_grid(lineup, lambda r: slot_by_id[r.name])
        if len(bench):
            ui.section("Bench")
            card_grid(bench, lambda r: "BENCH", ncols=4)
        with st.expander("Full numbers for your roster"):
            table(proj.loc[roster])

# ---------------------------------------------------------------- rankings
with tab_rank:
    c1, c2 = st.columns([3, 2])
    with c1:
        pos = st.segmented_control("Position", ["FLEX"] + model.POSITIONS, default="FLEX",
                                   format_func=lambda p: SLOT_LABELS.get(p, p))
    search = c2.text_input("Find a player", placeholder="Name or team…")
    view = proj[proj.opp.notna()]
    view = view[view.position.isin(["RB", "WR", "TE"] if pos in (None, "FLEX") else [pos])]
    if search:
        view = view[view.name.str.contains(search, case=False, regex=False) | view.team.str.fullmatch(search.upper())]
    table(view.head(80), height=720)
    st.caption("**Proj** = blended points per game (this season, with last season as a prior) × **Matchup** "
               "(opponent points allowed to the position) × **Scheme** (opponent efficiency vs pass/run) × "
               "**Vegas** (implied team total) × **Weather** (how the position scores in the forecast conditions). "
               "Green multipliers help, red ones hurt.  \n"
               f"**1st-read %, Catchable %, Contested %, Drops** come from {season} FTN charting"
               + (f" (through Week {ftn_through})" if ftn_through else "") + ": first-read share is how often the "
               "QB's first read was this player, out of the team's first-read throws in games they played.")

# ---------------------------------------------------------------- usage shifts
with tab_shift:
    shifts = get_usage_shifts(season, ppr, week)
    c1, c2 = st.columns([2, 3])
    with c1:
        spos = st.segmented_control("Position", ["All", "RB", "WR", "TE"], default="All", key="shift_pos") or "All"
    with c2:
        smetric = st.segmented_control("Share", ["All"] + list(model.SHIFT_METRICS), default="All",
                                       key="shift_metric") or "All"
    if shifts.empty:
        st.info("No usage changes of 8+ points yet — this needs at least 3 games played.")
    else:
        v = shifts
        if spos != "All":
            v = v[v.position == spos]
        if smetric != "All":
            v = v[v.metric == smetric]
        up, down = int((v.change > 0).sum()), int((v.change < 0).sum())
        st.caption(f"**{len(v)} changes** · {up} up · {down} down · biggest first")
        info_cols = ctx["info"].reindex(v.player_id)
        out = pd.DataFrame({
            "photo": [photo(r) for _, r in info_cols.assign(team=v.team.values).iterrows()],
            "Player": v.name.values, "Pos": v.position.values, "Team": v.team.values, "Share": v.metric.values,
            "Before": v.before.values, "Last 2": v["last"].values, "Change": v.change.values,
            "Why": v.reason.values})
        st.dataframe(out.style.map(ui.sign_color, subset=["Change"]).format(
                         {"Before": "{:.0%}", "Last 2": "{:.0%}", "Change": lambda x: f"{x * 100:+.0f} pts"}),
                     hide_index=True, width="stretch", height=min(40 + 35 * len(out), 760), column_config={
                         "photo": st.column_config.ImageColumn("", width="small"),
                         "Player": st.column_config.TextColumn("Player", width="medium"),
                         "Why": st.column_config.TextColumn("Why", width="large")})
    before_games = int(played_through) - 2 if played_through else 0
    st.caption(
        f"Share of team targets, carries and first-read throws over each team's **last 2 games** vs the rest of the "
        f"{season} season since the player joined the team. A game without a target or carry counts as zero, so "
        "injuries and benchings show up. Changes of 8+ points are listed; ones where the player never averaged 2+ "
        f"of that stat per game are skipped. First-read share comes from {season} FTN charting. \"Why\" points to "
        "the player's own missed games first, otherwise the teammate whose share moved most the other way."
        + (f"  \n⚠️ Only {before_games} earlier game{'s' * (before_games != 1)} to compare against so far — expect "
           "big swings until the sample grows." if 0 < before_games < 3 else ""))

# ---------------------------------------------------------------- game breakdown
with tab_game:
    if sched.empty:
        st.warning("No schedule found for this week.")
    else:
        games = sched[sched.home].reset_index()
        pick = st.selectbox("Game", games.index,
                            format_func=lambda i: f"{games.opp[i]} @ {games.team[i]}  ·  {games.gameday[i]}")
        home, away = games.team[pick], games.opp[pick]
        f = fc_by_team.get(home)
        g = wp[(wp.home == home) & (wp.away == away)].iloc[0] if len(wp) and ((wp.home == home) & (wp.away == away)).any() else None
        st.markdown(ui.game_banner(away, home, teams, sched, f, g), unsafe_allow_html=True)
        if g is not None:
            win_panel(g, away, home)
        if f is not None:
            if pd.notna(f.get("temp")):
                st.caption(f"Forecast: {f.summary}")
            for cond in [c for c in f.tags if c in weather.ADVERSE]:
                parts = [f"{t} {tsplits.at[(t, cond), 'ppg']:.1f} pts/gm ({tsplits.at[(t, cond), 'ppg_diff']:+.1f} vs norm, "
                         f"{tsplits.at[(t, cond), 'games']:.0f} gms)" for t in (away, home) if (t, cond) in tsplits.index]
                if parts:
                    st.caption(f"{ui.WX_ICON.get(cond, '')} In {cond.lower()} games since {weather.HISTORY_START}: "
                               + " · ".join(parts))
        side = st.segmented_control("Matchup", [f"{away} offense vs {home} D", f"{home} offense vs {away} D"],
                                    default=f"{away} offense vs {home} D")
        off, dfn = (home, away) if side and side.startswith(home) else (away, home)
        o = offense.loc[off] if off in offense.index else pd.Series(dtype=float)
        d = defense.loc[dfn]
        cur_tag = f"{season}" + (f", Wk 1–{ftn_through}" if ftn_through else "")
        legend = (f"**{season}** stats are this season's play-by-play and FTN charting"
                  + (f" (through Week {ftn_through})" if ftn_through else "") + ".")
        if pers_season != season:
            legend += (f" Stats marked **{pers_season}** (personnel groupings, man/zone, pressure, coverage and "
                       f"sub-packages) come from last season's NFL participation data — {season} participation "
                       "hasn't been published yet.")
        st.info(legend, icon="🗓️")
        mix_order = ["plays", "share", "pass_rate", "success_rate", "epa_per_play"]
        c1, c2 = st.columns(2, gap="large")
        with c1:
            ui.section(f"{off} offense · {season}")
            metric_grid([
                ("Plays / game", o.get("plays_per_game"), offense.plays_per_game.mean(), "num"),
                ("Neutral pass rate", o.get("neutral_pass_rate"), offense.neutral_pass_rate.mean(), "pct"),
                ("EPA / play", o.get("epa_per_play"), offense.epa_per_play.mean(), "epa"),
                ("Shotgun", o.get("shotgun_rate"), offense.shotgun_rate.mean(), "pct"),
                ("Motion", o.get("motion_rate"), offense.motion_rate.mean(), "pct"),
                ("Play-action", o.get("play_action_rate"), offense.play_action_rate.mean(), "pct"),
            ])
            st.caption(f"**Formation tendencies · {cur_tag}** (FTN charting)")
            metric_grid([
                ("Under center", o.get("under_center_rate"), offense.under_center_rate.mean(), "pct"),
                ("Empty backfield", o.get("empty_rate"), offense.empty_rate.mean(), "pct"),
                ("2+ backs", o.get("two_back_rate"), offense.two_back_rate.mean(), "pct"),
                ("Pistol", o.get("pistol_rate"), offense.pistol_rate.mean(), "pct"),
                ("Screens (of passes)", o.get("screen_rate"), offense.screen_rate.mean(), "pct"),
                ("RPO", o.get("rpo_rate"), offense.rpo_rate.mean(), "pct"),
            ])
            st.dataframe(model.alignment_mix(data, off, "offense"), width="stretch", column_config=PERSONNEL_COLS,
                         column_order=mix_order)
            st.caption(f"**Personnel groupings · {pers_season}** (NFL participation data)")
            mix = model.formation_mix(data, off, "offense").head(6)
            st.dataframe(mix, width="stretch", column_config=PERSONNEL_COLS, column_order=mix_order)
        with c2:
            ui.section(f"{dfn} defense")
            metric_grid([
                (f"Blitz rate ({season})", d.get("blitz_rate"), defense.blitz_rate.mean(), "pct"),
                (f"Men in box ({season})", d.get("avg_box"), defense.avg_box.mean(), "num"),
                (f"Man coverage ({pers_season})", d.get("man_rate"), defense.get("man_rate", pd.Series(dtype=float)).mean(), "pct"),
                (f"Pressure rate ({pers_season})", d.get("pressure_rate"),
                 defense.get("pressure_rate", pd.Series(dtype=float)).mean(), "pct"),
                (f"Deep yds / att allowed ({season})", d.get("deep_ypa_allowed"), defense.deep_ypa_allowed.mean(), "num"),
                (f"10+ yd runs allowed ({season})", d.get("explosive_run_rate"), defense.explosive_run_rate.mean(), "pct"),
            ])
            st.caption(f"**vs offensive formations · {cur_tag}** (FTN charting) — what {dfn} has allowed by look")
            st.dataframe(model.alignment_mix(data, dfn, "defense"), width="stretch", column_config=PERSONNEL_COLS,
                         column_order=mix_order)
            if "top_coverage" in defense and pd.notna(d.get("top_coverage")):
                st.caption(f"Most-used coverage ({pers_season}): **{d.top_coverage}**")
            st.caption(f"**Sub-packages · {pers_season}** (NFL participation data)")
            st.dataframe(model.formation_mix(data, dfn, "defense"), width="stretch", column_config=PERSONNEL_COLS,
                         column_order=mix_order)
        ui.section(f"What {dfn} allows (fantasy pts / game, rank 1 = most generous)")
        st.caption(f"{season} games, blended with {season - 1} as a prior while the sample is small.")
        pcols = st.columns(5)
        for col, p in zip(pcols, ["QB", "RB", "WR", "TE", "K"]):
            col.metric(p, f"{d[f'{p}_pts_allowed']:.1f}", f"#{d[f'{p}_rank']} of {len(defense)}", delta_color="off")
        ui.section(f"{off} players this week")
        table(proj[proj.team == off].head(12))

# ---------------------------------------------------------------- weather
with tab_wx:
    if fc.empty:
        st.warning("No games found for this week.")
    else:
        order = fc.assign(bad=fc.tags.map(lambda t: any(c in weather.ADVERSE for c in t))) \
                  .sort_values(["bad", "kickoff"], ascending=[False, True])
        rows = list(order.iterrows())
        for start in range(0, len(rows), 4):
            for col, (_, f) in zip(st.columns(4), rows[start:start + 4]):
                col.markdown(ui.weather_card(f, teams), unsafe_allow_html=True)
        st.caption("Forecasts from Open-Meteo for the 3 hours after kickoff, refreshed hourly. They appear about "
                   "2 weeks out and firm up 2–3 days before the game — check again before lineups lock.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section(f"Weather vs scoring since {weather.HISTORY_START}")
        st.bar_chart(league_wx.drop(index="Normal outdoor").vs_normal.rename("Total pts vs normal outdoor game"),
                     horizontal=True, color="#3DDC84", height=260)
        st.caption(" · ".join(f"{c}: {int(g)} games" for c, g in league_wx.games.items()))
    with c2:
        ui.section("Fantasy multiplier by position")
        st.dataframe(effects.mult.unstack().reindex(model.POSITIONS)
                     .rename(index={"DST": "D/ST"}, columns=lambda c: f"{ui.WX_ICON.get(c, '')} {c}")
                     .style.format("{:.2f}").background_gradient(cmap=ui.CMAP, vmin=0.88, vmax=1.12), width="stretch")
        st.caption("Applied to projections when the forecast matches. Based on the last two seasons and pulled "
                   "toward 1.00 when the sample is small.")

    ui.section("Team weather splits")
    cond = st.segmented_control("Condition", weather.CONDITIONS, default=weather.WINDY,
                                format_func=lambda c: f"{ui.WX_ICON.get(c, '')} {c}") or weather.WINDY
    t = tsplits.xs(cond, level="condition").sort_values("ppg_diff", ascending=False)
    t = t.assign(logo=[teams.logo(x) for x in t.index])[["logo", "games", "ppg", "ppg_diff", "allowed", "allowed_diff"]]
    st.dataframe(t.style.format({"ppg": "{:.1f}", "allowed": "{:.1f}", "ppg_diff": "{:+.1f}", "allowed_diff": "{:+.1f}"})
                 .background_gradient(subset=["ppg_diff"], cmap=ui.CMAP, vmin=-8, vmax=8)
                 .background_gradient(subset=["allowed_diff"], cmap=ui.CMAP.reversed(), vmin=-8, vmax=8),
                 width="stretch", height=560, column_config={
                     "logo": st.column_config.ImageColumn("", width="small"), "games": "Games", "ppg": "Pts / gm",
                     "ppg_diff": "Pts vs norm", "allowed": "Allowed / gm", "allowed_diff": "Allowed vs norm"})
    st.caption(f"“vs norm” compares with the same team's average in all games since {weather.HISTORY_START}. "
               "Fewer than ~5 games is mostly noise.")

# ---------------------------------------------------------------- defenses
with tab_def:
    cols = {"blitz_rate": "Blitz %", "avg_pass_rushers": "Rushers", "avg_box": "Box", "man_rate": "Man %",
            "pressure_rate": "Pressure %", "Nickel (5 DB)": "Nickel %", "Dime+ (6+ DB)": "Dime %",
            "top_coverage": "Top coverage", "pass_epa_allowed": "Pass EPA", "rush_epa_allowed": "Rush EPA",
            **{f"{p}_pts_allowed": f"{p} pts" for p in model.POSITIONS},
            "DST_pts_allowed": "D/ST pts (vs this offense)"}
    t = defense[[c for c in cols if c in defense]].rename(columns=cols)
    t.insert(0, "logo", [teams.logo(x) for x in t.index])
    pct = [c for c in ["Blitz %", "Man %", "Pressure %", "Nickel %", "Dime %"] if c in t]
    pts = [cols[f"{p}_pts_allowed"] for p in model.POSITIONS]
    st.dataframe(t.style.format({**{c: "{:.0%}" for c in pct}, "Rushers": "{:.2f}", "Box": "{:.2f}",
                                 "Pass EPA": "{:+.3f}", "Rush EPA": "{:+.3f}", **{c: "{:.1f}" for c in pts}}, na_rep="–")
                 .background_gradient(subset=pts, cmap=ui.CMAP),
                 width="stretch", height=1180, column_config={"logo": st.column_config.ImageColumn("", width="small")})
    st.caption(f"Blitz and box counts from {season} FTN charting. Man %, pressure, nickel/dime and coverage from "
               f"{pers_season} NFL participation data (the latest published). Green = more fantasy points allowed. "
               "The D/ST column is how many points opposing defenses score against this team's offense.")

# ---------------------------------------------------------------- model scorecard
with tab_score:
    st.markdown("How well has Matchup Lab projected this season? Each completed week is re-projected using **only "
                "plays from the weeks before it**, then compared with what actually happened — and with a simple "
                "baseline: each player's season average going into the week.")
    if not st.session_state.get("scorecard_on"):
        st.button("Run the backtest", type="primary", on_click=lambda: st.session_state.update(scorecard_on=True))
        st.caption("Takes a few seconds the first time; results are then cached.")
    else:
        bt, bt_log = get_backtest(season, ppr, played_through)
        if bt.empty:
            st.info("No completed weeks to score yet.")
        else:
            overall = model.scorecard(bt, None).iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Weeks scored", f"{bt.week.nunique()}", f"Weeks {bt.week.min()}–{bt.week.max()}",
                      delta_color="off")
            m2.metric("Avg miss (MAE)", f"{overall.mae:.2f} pts",
                      f"{overall.mae - overall.base_mae:+.2f} vs season avg ({overall.base_mae:.2f})",
                      delta_color="inverse", help="Mean absolute error per player-game, fantasy-relevant players.")
            m3.metric("Start/sit hit rate", f"{overall.start_hit:.0%}",
                      f"{(overall.start_hit - overall.base_start_hit) * 100:+.0f} pts vs season avg "
                      f"({overall.base_start_hit:.0%})", help="Of the players projected as starters (top 12 QB/TE/K/"
                      "D/ST, top 24 RB, top 36 WR each week), the share who finished as one.")
            m4.metric("Bias", f"{overall.bias:+.2f} pts", "projects high" if overall.bias > 0.5 else
                      "projects low" if overall.bias < -0.5 else "about even", delta_color="off",
                      help="Average of projection − actual. Positive means projections run high.")

            score_cols = {
                "players": st.column_config.NumberColumn("Player-games"),
                "mae": st.column_config.NumberColumn("Model MAE"),
                "base_mae": st.column_config.NumberColumn("Season-avg MAE"),
                "gain": st.column_config.NumberColumn("Error cut", help="How much smaller the model's average miss "
                                                      "is than the season-average baseline's."),
                "bias": st.column_config.NumberColumn("Bias"),
                "start_hit": st.column_config.NumberColumn("Start/sit (model)"),
                "base_start_hit": st.column_config.NumberColumn("Start/sit (season avg)"),
            }
            order = list(score_cols)

            def score_style(t: pd.DataFrame):
                t = t.rename_axis(None)
                return t.style.map(ui.sign_color, subset=["gain"]).format(
                    {"players": "{:.0f}", "mae": "{:.2f}", "base_mae": "{:.2f}", "gain": "{:+.0%}", "bias": "{:+.2f}",
                     "start_hit": "{:.0%}", "base_start_hit": "{:.0%}"}, na_rep="–")
            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                ui.section("By position")
                by_pos = model.scorecard(bt, "position").reindex(model.POSITIONS).rename(index={"DST": "D/ST"})
                st.dataframe(score_style(by_pos), width="stretch",
                             column_config=score_cols, column_order=order)
            with c2:
                ui.section("By week")
                by_week = model.scorecard(bt, "week").rename(index=lambda w: f"Week {w}")
                st.dataframe(score_style(by_week), width="stretch",
                             column_config=score_cols, column_order=["mae", "base_mae", "gain", "start_hit"])

            ui.section("Biggest misses")
            c1, c2 = st.columns([3, 2])
            with c1:
                mpos = st.segmented_control("Position", ["All"] + model.POSITIONS, default="All", key="miss_pos",
                                            format_func=lambda p: SLOT_LABELS.get(p, p)) or "All"
            with c2:
                mkind = st.segmented_control("Show", ["All misses", "Start/sit misses"], default="All misses",
                                             key="miss_kind") or "All misses"
            miss = bt[bt.relevant]
            if mpos != "All":
                miss = miss[miss.position == mpos]
            call = np.select([miss.start & ~miss.should_start, ~miss.start & miss.should_start],
                             ["Busted start", "Missed sleeper"], "")
            miss = miss.assign(call=call)
            if mkind == "Start/sit misses":
                miss = miss[miss.call != ""]
            miss = miss.reindex(miss.error.abs().sort_values(ascending=False).index).head(40)
            st.dataframe(pd.DataFrame({
                "Week": miss.week.values, "Player": miss.name.values, "Pos": miss.position.values,
                "Team": miss.team.values, "Opp": miss.opp.values, "Proj": miss.proj.values,
                "Season avg": miss.baseline.values, "Actual": miss.actual.values, "Miss": miss.error.values,
                "Start/sit": miss.call.values}).style.map(lambda v: ui.sign_color(-v), subset=["Miss"]).format(
                    {"Proj": "{:.1f}", "Season avg": "{:.1f}", "Actual": "{:.1f}", "Miss": "{:+.1f}"}),
                hide_index=True, width="stretch", height=min(40 + 35 * len(miss), 720))
            st.caption("**Miss** = projection − actual (negative: the player beat the projection). **Busted start** "
                       "= projected as a starter but finished outside the top group; **Missed sleeper** = the reverse.")

            with st.expander("How this is tested — and how we know it can't see the future"):
                st.markdown(
                    f"- For each completed week, current-season play-by-play, FTN charting and participation rows are "
                    f"cut to **earlier weeks only** before `project()` runs. Last season's data and each game's "
                    f"pregame Vegas lines are allowed — both were known before kickoff.\n"
                    "- Each player's team is the one they played for that week (known at kickoff), not today's "
                    "roster, and today's injury/roster status isn't used.\n"
                    "- Weather adjustments aren't applied here (past forecasts aren't stored), so these are the "
                    "projections before the weather multiplier.\n"
                    "- Scored on players who played that week and whom either method ranked inside twice the starter "
                    "count at their position — chosen with pregame info only, so neither method gets to pick.\n"
                    "- `tests/test_backtest_leakage.py` scrambles every play from the target week onward and checks "
                    "the projections don't change at all.")
                log = bt_log.assign(latest_week=bt_log.latest_week.map(
                    lambda w: "— (last season only)" if pd.isna(w) else f"Week {int(w)}"))
                st.dataframe(log.rename(columns={"week": "Week projected", "plays": f"{season} plays used",
                                                 "latest_week": "Latest week used", "ftn_rows": "FTN rows used",
                                                 "ftn_ok": "FTN rows all from earlier games"}),
                             hide_index=True, width="stretch")

st.caption("Data: [nflverse](https://github.com/nflverse) (play-by-play, participation, rosters, schedules & Vegas "
           "lines) · Charting data: [FTN Data](https://ftndata.com), licensed "
           "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) · Weather data by "
           "[Open-Meteo.com](https://open-meteo.com) (CC BY 4.0) · Scores, win probability, logos & headshots: ESPN / NFL.")
