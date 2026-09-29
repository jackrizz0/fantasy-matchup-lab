"""Import fantasy rosters, lineup slots and scoring from Sleeper, ESPN and Yahoo leagues.

Every importer returns a list of teams:
    {"team": str, "owner": str, "mine": bool, "player_ids": [app ids], "unmatched": [str],
     "slots": {"QB": 1, ...}, "ppr": float}
App ids are nflverse gsis ids for players and "DST_<TEAM>" for team defenses.
"""
from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

SLEEPER = "https://api.sleeper.app/v1"
ESPN = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{league}"

# Platform team abbreviations that differ from nflverse's.
TEAM_FIX = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX"}
ESPN_TEAMS = {1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET", 9: "GB", 10: "TEN",
              11: "IND", 12: "KC", 13: "LV", 14: "LA", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG",
              20: "NYJ", 21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS",
              29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU"}
ESPN_SLOTS = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 7: "SUPERFLEX", 16: "DST", 17: "K", 23: "FLEX"}
ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
SLEEPER_SLOTS = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DEF": "DST",
                 "FLEX": "FLEX", "SUPER_FLEX": "SUPERFLEX"}
YAHOO_SLOTS = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DEF": "DST", "W/R/T": "FLEX",
               "W/R": "FLEX", "W/T": "FLEX", "R/T": "FLEX", "Q/W/R/T": "SUPERFLEX"}


class LeagueImportError(Exception):
    """Raised with a message that is safe to show in the UI."""


def _get(url: str, **kw) -> dict | list:
    try:
        r = requests.get(url, timeout=20, **kw)
    except requests.RequestException as e:
        raise LeagueImportError(f"Couldn't reach the fantasy site: {e}") from e
    if r.status_code in (401, 403):
        raise LeagueImportError("This league is private. Add your espn_s2 and SWID cookies (see the help text).")
    if r.status_code == 404:
        raise LeagueImportError("League or user not found — double-check the ID/username and season.")
    r.raise_for_status()
    return r.json()


def id_maps(data: dict) -> dict:
    """Lookups from each platform's player id (and name+position) to the app's gsis id."""
    r = pd.concat([x for x in (data["roster"], data["roster_prev"]) if x is not None]).drop_duplicates("gsis_id")
    r = r[r.gsis_id.notna()]
    out = {}
    for col in ("sleeper_id", "espn_id", "yahoo_id"):
        s = r.dropna(subset=[col])
        out[col] = dict(zip(s[col].astype(str).str.replace(r"\.0$", "", regex=True), s.gsis_id))
    out["name"] = {(_norm(n), p): g for n, p, g in zip(r.full_name, r.position, r.gsis_id)}
    return out


def _norm(name: str) -> str:
    """'D.K. Metcalf Jr.' -> 'dk metcalf' so names match across sites."""
    name = re.sub(r"[.'’]", "", str(name).lower())
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return " ".join(name.split())


def _ppr(value) -> float:
    try:
        return min((0.0, 0.5, 1.0), key=lambda x: abs(x - float(value)))
    except (TypeError, ValueError):
        return 1.0


# ---------------------------------------------------------------- Sleeper
def sleeper_user_id(username: str) -> str:
    user = _get(f"{SLEEPER}/user/{username.strip()}")
    if not user:
        raise LeagueImportError(f"No Sleeper user named '{username}'.")
    return user["user_id"]


def sleeper_leagues(user_id: str, season: int) -> list[dict]:
    leagues = _get(f"{SLEEPER}/user/{user_id}/leagues/nfl/{season}") or []
    return [{"league_id": lg["league_id"], "name": lg["name"]} for lg in leagues]


def sleeper_teams(league_id: str, maps: dict, user_id: str | None = None) -> list[dict]:
    league = _get(f"{SLEEPER}/league/{league_id.strip()}")
    if not league:
        raise LeagueImportError("League not found — double-check the Sleeper league ID.")
    rosters = _get(f"{SLEEPER}/league/{league_id}/rosters")
    users = {u["user_id"]: u for u in _get(f"{SLEEPER}/league/{league_id}/users")}

    slots = {}
    for pos in league.get("roster_positions", []):
        if pos in SLEEPER_SLOTS:
            slots[SLEEPER_SLOTS[pos]] = slots.get(SLEEPER_SLOTS[pos], 0) + 1
    ppr = _ppr(league.get("scoring_settings", {}).get("rec", 1))

    teams = []
    for r in rosters:
        owner = users.get(r.get("owner_id"), {})
        ids, unmatched = [], []
        for pid in r.get("players") or []:
            if pid.isalpha():  # team defenses are stored by team abbreviation
                ids.append(f"DST_{TEAM_FIX.get(pid, pid)}")
            elif pid in maps["sleeper_id"]:
                ids.append(maps["sleeper_id"][pid])
            else:
                unmatched.append(f"Sleeper player {pid}")
        teams.append({
            "team": (owner.get("metadata") or {}).get("team_name") or owner.get("display_name") or f"Roster {r['roster_id']}",
            "owner": owner.get("display_name", ""),
            "mine": bool(user_id) and r.get("owner_id") == user_id,
            "player_ids": ids, "unmatched": unmatched, "slots": slots, "ppr": ppr,
        })
    return teams


# ---------------------------------------------------------------- ESPN
def espn_teams(league_id: str, season: int, maps: dict, espn_s2: str = "", swid: str = "") -> list[dict]:
    cookies = {"espn_s2": espn_s2.strip(), "SWID": swid.strip()} if espn_s2 and swid else None
    url = ESPN.format(season=season, league=league_id.strip())
    league = _get(url, params=[("view", "mTeam"), ("view", "mRoster"), ("view", "mSettings")], cookies=cookies)

    settings = league.get("settings", {})
    counts = settings.get("rosterSettings", {}).get("lineupSlotCounts", {})
    slots = {}
    for sid, n in counts.items():
        name = ESPN_SLOTS.get(int(sid))
        if name and n:
            slots[name] = slots.get(name, 0) + int(n)
    rec = next((i.get("points") for i in settings.get("scoringSettings", {}).get("scoringItems", [])
                if i.get("statId") == 53), 0)
    ppr = _ppr(rec)

    members = {m["id"]: m for m in league.get("members", [])}
    teams = []
    for t in league.get("teams", []):
        ids, unmatched = [], []
        for e in t.get("roster", {}).get("entries", []):
            p = e.get("playerPoolEntry", {}).get("player", {})
            pos = ESPN_POS.get(p.get("defaultPositionId"))
            if pos == "DST":
                team = ESPN_TEAMS.get(p.get("proTeamId"))
                (ids if team else unmatched).append(f"DST_{team}" if team else p.get("fullName", "?"))
                continue
            gid = maps["espn_id"].get(str(p.get("id"))) or maps["name"].get((_norm(p.get("fullName")), pos))
            (ids if gid else unmatched).append(gid or p.get("fullName", "?"))
        owners = [members.get(o, {}) for o in t.get("owners", [])]
        owner = " & ".join(f"{m.get('firstName', '')} {m.get('lastName', '')}".strip() for m in owners)
        name = t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip() or f"Team {t.get('id')}"
        teams.append({
            "team": name, "owner": owner,
            "mine": bool(swid) and swid.strip().upper() in [o.upper() for o in t.get("owners", [])],
            "player_ids": ids, "unmatched": unmatched, "slots": slots, "ppr": ppr,
        })
    return teams


# ---------------------------------------------------------------- Yahoo
# Yahoo requires OAuth: the user registers a free app at https://developer.yahoo.com/apps/create
# (Fantasy Sports: Read, redirect URI "oob"), then pastes the code Yahoo shows after authorizing.
YAHOO_AUTH = "https://api.login.yahoo.com/oauth2"
YAHOO_API = "https://fantasysports.yahooapis.com/fantasy/v2"
YAHOO_TOKEN_FILE = Path(__file__).parent / ".yahoo_auth.json"


def yahoo_auth_url(client_id: str) -> str:
    return f"{YAHOO_AUTH}/request_auth?client_id={client_id.strip()}&redirect_uri=oob&response_type=code"


def _yahoo_token_request(client_id: str, client_secret: str, **form) -> dict:
    try:
        r = requests.post(f"{YAHOO_AUTH}/get_token", auth=(client_id.strip(), client_secret.strip()),
                          data={"redirect_uri": "oob", **form}, timeout=20)
    except requests.RequestException as e:
        raise LeagueImportError(f"Couldn't reach Yahoo: {e}") from e
    if not r.ok:
        raise LeagueImportError(f"Yahoo sign-in failed ({r.status_code}). Check the Client ID/Secret, "
                                "or get a fresh code — each code works only once.")
    tok = r.json()
    saved = {"client_id": client_id.strip(), "client_secret": client_secret.strip(),
             "access_token": tok["access_token"], "refresh_token": tok.get("refresh_token", form.get("refresh_token")),
             "expires_at": time.time() + int(tok.get("expires_in", 3600)) - 60}
    YAHOO_TOKEN_FILE.write_text(json.dumps(saved))
    os.chmod(YAHOO_TOKEN_FILE, 0o600)  # readable only by you
    return saved


def yahoo_connect(client_id: str, client_secret: str, code: str) -> None:
    _yahoo_token_request(client_id, client_secret, grant_type="authorization_code", code=code.strip())


def yahoo_connected() -> bool:
    return YAHOO_TOKEN_FILE.exists()


def yahoo_disconnect() -> None:
    YAHOO_TOKEN_FILE.unlink(missing_ok=True)


def _yahoo_access_token() -> str:
    if not YAHOO_TOKEN_FILE.exists():
        raise LeagueImportError("Connect your Yahoo account first.")
    tok = json.loads(YAHOO_TOKEN_FILE.read_text())
    if time.time() > tok["expires_at"]:
        tok = _yahoo_token_request(tok["client_id"], tok["client_secret"],
                                   grant_type="refresh_token", refresh_token=tok["refresh_token"])
    return tok["access_token"]


def _yahoo_get(path: str) -> ET.Element:
    try:
        r = requests.get(f"{YAHOO_API}/{path}", headers={"Authorization": f"Bearer {_yahoo_access_token()}"}, timeout=20)
    except requests.RequestException as e:
        raise LeagueImportError(f"Couldn't reach Yahoo: {e}") from e
    if r.status_code == 401:
        raise LeagueImportError("Yahoo rejected the saved sign-in. Disconnect and connect again.")
    if not r.ok:
        raise LeagueImportError(f"Yahoo returned an error ({r.status_code}).")
    root = ET.fromstring(r.content)
    for el in root.iter():  # drop the XML namespace so paths stay readable
        el.tag = el.tag.split("}", 1)[-1]
    return root


def yahoo_leagues(season: int) -> list[dict]:
    root = _yahoo_get("users;use_login=1/games;game_codes=nfl;seasons=%d/leagues" % season)
    return [{"league_key": lg.findtext("league_key"), "name": lg.findtext("name")} for lg in root.iter("league")]


def yahoo_teams(league_key: str, maps: dict) -> list[dict]:
    settings = _yahoo_get(f"league/{league_key}/settings")
    slots = {}
    for rp in settings.iter("roster_position"):
        name = YAHOO_SLOTS.get(rp.findtext("position"))
        if name:
            slots[name] = slots.get(name, 0) + int(rp.findtext("count") or 0)
    rec = next((s.findtext("value") for s in settings.iter("stat")
                if s.findtext("stat_id") == "11" and s.findtext("value") is not None), 0)  # 11 = receptions
    ppr = _ppr(rec)

    root = _yahoo_get(f"league/{league_key}/teams/roster")
    teams = []
    for t in root.iter("team"):
        ids, unmatched = [], []
        for p in t.iter("player"):
            pos = p.findtext("display_position") or ""
            name = p.findtext("name/full") or "?"
            if pos == "DEF":
                abbr = (p.findtext("editorial_team_abbr") or "").upper()
                ids.append(f"DST_{TEAM_FIX.get(abbr, abbr)}")
                continue
            main_pos = pos.split(",")[0]
            gid = maps["yahoo_id"].get(p.findtext("player_id") or "") or maps["name"].get((_norm(name), main_pos))
            (ids if gid else unmatched).append(gid or name)
        teams.append({
            "team": t.findtext("name") or t.findtext("team_key"),
            "owner": ", ".join(m.findtext("nickname") or "" for m in t.iter("manager")),
            "mine": t.findtext("is_owned_by_current_login") == "1",
            "player_ids": ids, "unmatched": unmatched, "slots": slots, "ppr": ppr,
        })
    return teams
