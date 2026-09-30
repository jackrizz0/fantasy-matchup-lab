"""Visual building blocks: page CSS, header, live scoreboard, player cards, game cards and table styling."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

import weather

# Diverging scale that sits on the dark theme: muted red -> card surface -> turf green.
CMAP = LinearSegmentedColormap.from_list("ml", ["#7a2633", "#131B2E", "#1d7a4c"])
GOOD, BAD, MUTED, ACCENT = "#3DDC84", "#FF6B6B", "#8A97B1", "#3DDC84"
WX_ICON = {weather.WINDY: "💨", weather.WET: "🌧️", weather.COLD: "🥶", weather.HOT: "🔥", weather.INDOORS: "🏟️"}
SOURCE_LABEL = {"live": "Live · ESPN", "final": "Final", "market": "Betting market", "model": "ESPN predictor"}

CSS = """
<style>
  .block-container { padding-top: 1.6rem; max-width: 1320px; }
  [data-testid="stMetricValue"], .ml-num { font-variant-numeric: tabular-nums; }
  @keyframes ml-pulse { 0% { box-shadow: 0 0 0 0 rgba(255,77,94,.7); } 70% { box-shadow: 0 0 0 7px rgba(255,77,94,0); }
                        100% { box-shadow: 0 0 0 0 rgba(255,77,94,0); } }
  @keyframes ml-rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

  /* ---------- hero ---------- */
  .ml-hero { position: relative; overflow: hidden; border-radius: 20px; padding: 26px 30px 22px; margin-bottom: 14px;
    background: radial-gradient(1200px 300px at 85% -40%, rgba(61,220,132,.28), transparent 60%),
                radial-gradient(700px 260px at 0% 120%, rgba(111,195,255,.20), transparent 60%),
                linear-gradient(135deg, #111c36 0%, #0c1426 100%);
    border: 1px solid rgba(255,255,255,.08); }
  .ml-hero::before { content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .5;
    background: repeating-linear-gradient(90deg, transparent 0 88px, rgba(255,255,255,.035) 88px 90px); }
  .ml-hero .row { position: relative; display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; flex-wrap: wrap; }
  .ml-kicker { color: #3DDC84; font-weight: 700; font-size: .75rem; letter-spacing: .22em; text-transform: uppercase; }
  .ml-hero h1 { font-family: "Barlow Condensed", sans-serif; font-size: 3.1rem; line-height: .95; margin: 6px 0 8px; padding: 0;
    letter-spacing: .5px; background: linear-gradient(90deg, #fff 0%, #cfe9ff 55%, #3DDC84 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent; }
  .ml-hero .sub { color: #AEB9D0; font-size: .95rem; }
  .ml-big { text-align: right; }
  .ml-big .n { font-family: "Barlow Condensed", sans-serif; font-size: 4.2rem; font-weight: 700; line-height: .85; color: #E7ECF5; }
  .ml-big .l { color: #8A97B1; font-size: .72rem; letter-spacing: .18em; text-transform: uppercase; }
  .ml-pills { position: relative; display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
  .ml-pill { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1); border-radius: 999px;
    padding: 4px 12px; font-size: .8rem; color: #D5DCEA; white-space: nowrap; backdrop-filter: blur(6px); }

  /* ---------- live scoreboard strip ---------- */
  .ml-strip { display: flex; gap: 10px; overflow-x: auto; padding: 2px 2px 10px; margin-bottom: 8px; scrollbar-width: thin; }
  .ml-tile { flex: 0 0 188px; background: #111a2d; border: 1px solid #22304D; border-radius: 14px; padding: 10px 12px 11px;
    animation: ml-rise .35s ease both; }
  .ml-tile.live { border-color: rgba(255,77,94,.55); background: linear-gradient(180deg, rgba(255,77,94,.08), #111a2d 45%); }
  .ml-tile .st { display: flex; justify-content: space-between; align-items: center; font-size: .7rem; color: #8A97B1;
    letter-spacing: .06em; text-transform: uppercase; margin-bottom: 6px; }
  .ml-live { color: #FF6B78; font-weight: 700; display: inline-flex; align-items: center; gap: 6px; }
  .ml-live i { width: 7px; height: 7px; border-radius: 50%; background: #FF4D5E; display: inline-block; animation: ml-pulse 1.6s infinite; }
  .ml-tile .tm { display: flex; align-items: center; gap: 7px; font-weight: 600; font-size: .9rem; margin: 3px 0; }
  .ml-tile .tm img { width: 20px; height: 20px; }
  .ml-tile .tm .sc { margin-left: auto; font-family: "Barlow Condensed", sans-serif; font-size: 1.15rem; font-weight: 700; }
  .ml-tile .tm .pc { margin-left: auto; color: #AEB9D0; font-size: .82rem; font-variant-numeric: tabular-nums; }
  .ml-tile .tm.fav .pc { color: #E7ECF5; font-weight: 700; }
  .ml-tile .poss { color: #FFC857; font-size: .7rem; margin-left: 2px; }
  .ml-split { display: flex; height: 6px; border-radius: 99px; overflow: hidden; margin-top: 8px; background: #22304D; }
  .ml-split span { display: block; height: 100%; transition: width .6s ease; }
  .ml-tile .src { color: #6B7894; font-size: .66rem; margin-top: 5px; }

  /* ---------- tabs & metrics ---------- */
  [data-baseweb="tab-list"] { gap: 6px; background: #0f1729; padding: 5px; border-radius: 14px; border: 1px solid #1c2842; }
  [data-baseweb="tab"] { border-radius: 10px !important; padding: 8px 14px !important; height: auto !important; }
  [data-baseweb="tab"][aria-selected="true"] { background: linear-gradient(135deg, #3DDC84, #2bb8a0) !important; }
  [data-baseweb="tab"][aria-selected="true"] p { color: #06110b !important; font-weight: 700; }
  [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
  [data-testid="stMetric"] { background: linear-gradient(180deg, #131B2E, #101727); border: 1px solid #22304D;
    border-radius: 14px; padding: 12px 16px; }
  [data-testid="stMetricValue"] { font-family: "Barlow Condensed", sans-serif; font-weight: 700; }
  [data-testid="stMetricDelta"] svg { display: none; }  /* deltas here are comparisons, not ups/downs */
  [data-testid="stExpander"] details { border-radius: 12px; }
  [data-baseweb="tag"] { background: #1a2842 !important; border: 1px solid #2A3858; }
  [data-baseweb="tag"] span { color: #DCE3F0 !important; }

  /* ---------- player cards ---------- */
  .ml-card { position: relative; overflow: hidden; background: #111a2d; border: 1px solid #22304D; border-radius: 16px;
    padding: 12px 14px 12px; margin-bottom: 4px; animation: ml-rise .35s ease both; }
  .ml-card::before { content: ""; position: absolute; inset: 0; pointer-events: none;
    background: radial-gradient(260px 120px at 100% 0%, color-mix(in srgb, var(--team) 45%, transparent), transparent 70%); }
  .ml-card::after { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--team); }
  .ml-card > * { position: relative; }
  .ml-card .hdr { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .ml-slot { font-size: .66rem; font-weight: 800; letter-spacing: .1em; color: #06110b; background: #3DDC84;
    border-radius: 6px; padding: 2px 8px; }
  .ml-slot.bench { background: #2A3858; color: #C9D2E3; }
  .ml-when { color: #8A97B1; font-size: .72rem; }
  .ml-card .top { display: flex; align-items: center; gap: 12px; }
  .ml-head { width: 64px; height: 64px; border-radius: 50%; object-fit: cover; flex: none;
    background: radial-gradient(circle at 50% 35%, color-mix(in srgb, var(--team) 70%, #fff 10%), #0b1222);
    border: 2px solid color-mix(in srgb, var(--team) 80%, #fff 20%); }
  .ml-name { font-weight: 700; font-size: 1.05rem; line-height: 1.15; }
  .ml-meta { color: #9AA6BF; font-size: .8rem; margin-top: 3px; display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
  .ml-meta img { width: 16px; height: 16px; }
  .ml-proj { margin-left: auto; text-align: right; flex: none; }
  .ml-proj .v { font-family: "Barlow Condensed", sans-serif; font-size: 2.3rem; font-weight: 700; line-height: .9; color: #3DDC84; }
  .ml-proj .l { color: #8A97B1; font-size: .62rem; text-transform: uppercase; letter-spacing: .12em; }
  .ml-win { margin-top: 10px; }
  .ml-win .lbl { display: flex; justify-content: space-between; color: #9AA6BF; font-size: .72rem; margin-bottom: 4px; }
  .ml-win .lbl b { color: #E7ECF5; }
  .ml-bar { height: 5px; background: #22304D; border-radius: 99px; overflow: hidden; }
  .ml-bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--team), #3DDC84); border-radius: 99px; }
  .ml-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
  .ml-chip { font-size: .72rem; border-radius: 999px; padding: 2px 9px; border: 1px solid #2A3858; color: #C9D2E3;
    background: rgba(23,34,57,.8); white-space: nowrap; }
  .ml-chip.good { color: #3DDC84; border-color: rgba(61,220,132,.35); background: rgba(61,220,132,.08); }
  .ml-chip.bad { color: #FF8A8A; border-color: rgba(255,107,107,.35); background: rgba(255,107,107,.08); }

  /* ---------- game banner ---------- */
  .ml-game { background: linear-gradient(90deg, color-mix(in srgb, var(--away) 22%, #111a2d), #111a2d 35%, #111a2d 65%,
    color-mix(in srgb, var(--home) 22%, #111a2d)); border: 1px solid #22304D; border-radius: 18px; padding: 18px 22px;
    margin: 6px 0 12px; }
  .ml-game .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .ml-team { display: flex; align-items: center; gap: 12px; }
  .ml-team.r { flex-direction: row-reverse; text-align: right; }
  .ml-team img { width: 58px; height: 58px; }
  .ml-team .n { font-family: "Barlow Condensed", sans-serif; font-size: 1.8rem; font-weight: 700; line-height: 1; }
  .ml-team .s { color: #9AA6BF; font-size: .8rem; }
  .ml-team .p { font-family: "Barlow Condensed", sans-serif; font-size: 2.4rem; font-weight: 700; line-height: 1; margin: 0 8px; }
  .ml-center { text-align: center; color: #AEB9D0; font-size: .82rem; line-height: 1.5; }
  .ml-at { color: #6B7894; font-family: "Barlow Condensed", sans-serif; font-size: 1.4rem; }
  .ml-game .ml-split { height: 8px; margin-top: 14px; }

  /* ---------- weather cards ---------- */
  .ml-wx { background: #111a2d; border: 1px solid #22304D; border-radius: 16px; padding: 12px 14px; margin-bottom: 12px;
    animation: ml-rise .35s ease both; }
  .ml-wx.alert { border-color: rgba(255,200,87,.5); background: linear-gradient(180deg, rgba(255,200,87,.07), #111a2d 50%); }
  .ml-wx .teams { display: flex; align-items: center; gap: 6px; font-weight: 600; }
  .ml-wx .teams img { width: 22px; height: 22px; }
  .ml-wx .when { color: #8A97B1; font-size: .76rem; margin-top: 2px; }
  .ml-wx .temp { font-family: "Barlow Condensed", sans-serif; font-size: 2.2rem; font-weight: 700; line-height: 1; margin-top: 10px; }
  .ml-wx .det { color: #AEB9D0; font-size: .8rem; margin-top: 4px; }

  .ml-note { color: #C9D2E3; font-size: .86rem; margin: 3px 0; }
  .ml-section { font-family: "Barlow Condensed", sans-serif; font-size: 1.45rem; font-weight: 700; margin: 20px 0 8px;
    letter-spacing: .3px; display: flex; align-items: center; gap: 10px; }
  .ml-section::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, #22304D, transparent); }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="ml-section">{escape(title)}</div>', unsafe_allow_html=True)


def hero(kicker: str, title: str, subtitle: str, big: str, big_label: str, pills: list[str]):
    pills_html = "".join(f'<span class="ml-pill">{escape(p)}</span>' for p in pills)
    st.markdown(
        f'<div class="ml-hero"><div class="row"><div><div class="ml-kicker">{escape(kicker)}</div>'
        f'<h1>{escape(title)}</h1><div class="sub">{escape(subtitle)}</div></div>'
        f'<div class="ml-big"><div class="n">{escape(big)}</div><div class="l">{escape(big_label)}</div></div></div>'
        f'<div class="ml-pills">{pills_html}</div></div>', unsafe_allow_html=True)


def _luminance(hex_color: str) -> float:
    try:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except (ValueError, AttributeError):
        return 1.0
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


class Teams:
    """Logo/color/name lookups with safe fallbacks."""

    def __init__(self, teams: pd.DataFrame | None):
        self.t = teams if teams is not None else pd.DataFrame()

    def _get(self, abbr, col, default=""):
        if abbr in self.t.index and col in self.t and pd.notna(self.t.at[abbr, col]):
            return self.t.at[abbr, col]
        return default

    def logo(self, abbr):
        return self._get(abbr, "team_logo_espn")

    def color(self, abbr):
        """Primary team color, or the secondary one when the primary would vanish on the dark background."""
        main = self._get(abbr, "team_color", "#3DDC84")
        if _luminance(main) < 0.06:
            alt = self._get(abbr, "team_color2", main)
            if _luminance(alt) > _luminance(main):
                return alt
        return main

    def name(self, abbr):
        return self._get(abbr, "team_name", abbr)


def _chip(text: str, tone: str = "") -> str:
    return f'<span class="ml-chip {tone}">{escape(text)}</span>'


def _tone(mult, hi=1.04, lo=0.96) -> str:
    if pd.isna(mult):
        return ""
    return "good" if mult >= hi else "bad" if mult <= lo else ""


def _strftime(ts, fmt: str) -> str:
    """strftime that also understands %-d / %-m / %-I (no leading zero) on Windows, not just Linux/macOS."""
    for code in "dmI":
        fmt = fmt.replace(f"%-{code}", str(int(ts.strftime(f"%{code}"))))
    return ts.strftime(fmt)


def pct(p: float, live: bool = False) -> str:
    """Percent label; while a game is still going, never claim 0% or 100%."""
    if live:
        p = min(max(p, 0.01), 0.99)
    return f"{p:.0%}"


def _split_bar(away_p: float, away_color: str, home_color: str) -> str:
    a = max(0.0, min(1.0, away_p)) * 100
    return (f'<div class="ml-split"><span style="width:{a:.1f}%;background:{away_color}"></span>'
            f'<span style="width:{100 - a:.1f}%;background:{home_color}"></span></div>')


# ---------------------------------------------------------------- live scoreboard
def scoreboard_strip(wp: pd.DataFrame, teams: Teams) -> str:
    if wp is None or wp.empty:
        return ""
    order = {"in": 0, "pre": 1, "post": 2}
    tiles = []
    for _, g in wp.sort_values(["state", "kickoff"], key=lambda s: s.map(order) if s.name == "state" else s).iterrows():
        live = g.state == "in"
        if live:
            status = f'<span class="ml-live"><i></i>LIVE</span><span>{escape(g.detail)}</span>'
        elif g.state == "post":
            status = f"<span>{escape(g.detail)}</span>"
        else:
            status = f'<span>{_strftime(g.kickoff, "%a %-I:%M %p")}</span>'
        hp = g.home_win if g.home_win is not None and pd.notna(g.home_win) else None
        rows = ""
        for side, score, p in (("away", g.away_score, None if hp is None else 1 - hp), ("home", g.home_score, hp)):
            t = g[side]
            poss = '<span class="poss">●</span>' if live and g.possession == t else ""
            right = (f'<span class="sc">{score}</span>' if g.state != "pre"
                     else f'<span class="pc">{pct(p)}</span>' if p is not None else "")
            if g.state == "in" and p is not None:
                right = (f'<span class="pc" style="margin-left:auto;margin-right:8px">{pct(p, True)}</span>'
                         f'<span class="sc" style="margin-left:0">{score}</span>')
            fav = " fav" if p is not None and p > 0.5 else ""
            rows += (f'<div class="tm{fav}"><img src="{teams.logo(t)}" alt="">{escape(t)}{poss}{right}</div>')
        bar = _split_bar(1 - hp, teams.color(g.away), teams.color(g.home)) if hp is not None else ""
        src = f'<div class="src">{SOURCE_LABEL.get(g.source, "")}' + (
            f" · {escape(g.down_distance)}" if live and g.down_distance else "") + "</div>"
        tiles.append(f'<div class="ml-tile{" live" if live else ""}"><div class="st">{status}</div>{rows}{bar}{src}</div>')
    return f'<div class="ml-strip">{"".join(tiles)}</div>'


# ---------------------------------------------------------------- player cards
def player_card(r: pd.Series, slot: str | None, teams: Teams, fc_row, defense: pd.DataFrame) -> str:
    color = teams.color(r.team)
    photo = teams.logo(r.team) if r.position == "DST" or not isinstance(r.get("headshot"), str) else r.headshot
    bye = pd.isna(r.opp)
    meta = f'{escape(str(r.team))} · {escape(r.position)}'
    if not bye:
        meta += f' · vs <img src="{teams.logo(r.opp)}" alt=""> {escape(str(r.opp))}'
    when = "BYE WEEK" if bye else (_strftime(fc_row.kickoff, "%a %-I:%M %p") if fc_row is not None else "")
    chips = []
    if not bye and r.opp in defense.index:
        rank = defense.at[r.opp, f"{r.position}_rank"]
        label = "vs offense" if r.position == "DST" else f"vs {r.position}"
        chips.append(_chip(f"#{rank} matchup {label}", _tone(r.dvp_mult)))
    if pd.notna(r.get("implied_total")):
        chips.append(_chip(f"Team total {r.implied_total:.1f}", _tone(r.env_mult)))
    if fc_row is not None and fc_row.tags:
        for tag in fc_row.tags:
            tone = "" if tag == weather.INDOORS else _tone(r.weather_mult, 1.01, 0.99)
            chips.append(_chip(f"{WX_ICON.get(tag, '')} {tag}", tone))
    elif fc_row is not None and pd.notna(fc_row.get("temp")):
        chips.append(_chip(f"☀️ {fc_row.temp:.0f}°F · {fc_row.wind:.0f} mph"))
    win = ""
    if pd.notna(r.get("win_prob")):
        win = (f'<div class="ml-win"><div class="lbl"><span>{escape(str(r.team))} win chance</span>'
               f'<span><b>{pct(r.win_prob, r.get("win_source") == "live")}</b> · {escape(SOURCE_LABEL.get(r.get("win_source"), ""))}</span></div>'
               f'<div class="ml-bar"><span style="width:{r.win_prob * 100:.1f}%"></span></div></div>')
    slot_html = f'<span class="ml-slot{" bench" if slot == "BENCH" else ""}">{escape(slot or "")}</span>'
    img = f'<img class="ml-head" src="{photo}" alt="">' if photo else '<div class="ml-head"></div>'
    return (f'<div class="ml-card" style="--team:{color}"><div class="hdr">{slot_html}<span class="ml-when">{when}</span></div>'
            f'<div class="top">{img}<div style="min-width:0"><div class="ml-name">{escape(str(r["name"]))}</div>'
            f'<div class="ml-meta">{meta}</div></div>'
            f'<div class="ml-proj"><div class="v ml-num">{r.proj:.1f}</div><div class="l">proj pts</div></div></div>'
            f'{win}<div class="ml-chips">{"".join(chips)}</div></div>')


def game_banner(away: str, home: str, teams: Teams, sched: pd.DataFrame, fc_row, g=None) -> str:
    """Matchup header; `g` is the win-probability row for the game (score, status, win %)."""
    started = g is not None and g.state != "pre"
    hp = g.home_win if g is not None and g.home_win is not None and pd.notna(g.home_win) else None

    def side(t, cls, p, score):
        it = sched.implied_total.get(t) if len(sched) else None
        sub = f"Implied {it:.1f}" if it is not None and pd.notna(it) else teams.name(t)
        if p is not None and g is not None and g.state != "post":
            sub += f" · {pct(p, g.state == 'in')} to win"
        big = f'<div class="p">{score}</div>' if started else ""
        return (f'<div class="ml-team {cls}"><img src="{teams.logo(t)}" alt=""><div><div class="n">{escape(t)}</div>'
                f'<div class="s">{escape(sub)}</div></div>{big}</div>')

    center = []
    if g is not None and g.state == "in":
        center.append(f'<span class="ml-live"><i></i>LIVE</span> {escape(g.detail)}')
        if g.down_distance:
            center.append(escape(g.down_distance))
    elif g is not None and g.state == "post":
        center.append(escape(g.detail))
    elif fc_row is not None:
        center.append(_strftime(fc_row.kickoff, "%a %-m/%-d · %-I:%M %p ET"))
    if fc_row is not None:
        center.append(escape(str(fc_row.stadium)))
        if fc_row.tags:
            center.append(" ".join(f"{WX_ICON.get(t, '')} {escape(t)}" for t in fc_row.tags))
    bar = _split_bar(1 - hp, teams.color(away), teams.color(home)) if hp is not None else ""
    return (f'<div class="ml-game" style="--away:{teams.color(away)};--home:{teams.color(home)}"><div class="row">'
            f'{side(away, "", None if hp is None else 1 - hp, g.away_score if g is not None else "")}'
            f'<div class="ml-center"><div class="ml-at">@</div>{"<br>".join(center)}</div>'
            f'{side(home, "r", hp, g.home_score if g is not None else "")}</div>{bar}</div>')


def weather_card(f: pd.Series, teams: Teams) -> str:
    adverse = [t for t in f.tags if t in weather.ADVERSE]
    when = _strftime(f.kickoff, "%a %-I:%M %p ET")
    if weather.INDOORS in f.tags:
        big, det = "🏟️ Indoors", f"{escape(str(f.roof).title())} roof — no weather impact"
    elif pd.notna(f.get("temp")):
        icon = WX_ICON.get(adverse[0], "☀️") if adverse else "☀️"
        big = f"{icon} {f.temp:.0f}°F"
        det = f"💨 {f.wind:.0f} mph (gusts {f.gusts:.0f}) · 💧 {f.precip_prob:.0f}%"
        if f.precip > 0:
            det += f" · {f.precip:.2f}\""
    else:
        big, det = "—", escape(str(f.summary))
    chips = "".join(_chip(f"{WX_ICON.get(t, '')} {t}", "bad") for t in adverse)
    return (f'<div class="ml-wx{" alert" if adverse else ""}"><div class="teams">'
            f'<img src="{teams.logo(f.away_team)}" alt="">{escape(f.away_team)} @ '
            f'<img src="{teams.logo(f.home_team)}" alt="">{escape(f.home_team)}</div>'
            f'<div class="when">{when} · {escape(str(f.stadium))}</div><div class="temp">{big}</div>'
            f'<div class="det">{det}</div><div class="ml-chips">{chips}</div></div>')


def mult_color(v):
    if pd.isna(v):
        return ""
    return f"color: {GOOD}" if v >= 1.03 else f"color: {BAD}" if v <= 0.97 else f"color: {MUTED}"
