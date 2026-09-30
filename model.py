"""Fantasy scoring, opponent scheme tendencies, matchup projections and lineup optimization."""
from __future__ import annotations

import numpy as np
import pandas as pd

SKILL = ["QB", "RB", "WR", "TE"]
POSITIONS = SKILL + ["K", "DST"]
# Standard D/ST points-allowed tiers: (max points allowed, fantasy points)
PA_TIERS = [(0, 10), (6, 7), (13, 4), (20, 1), (27, 0), (34, -1), (999, -4)]
PRIOR_GAMES = 3        # how many games' worth of weight last season's average gets for a player
DST_PRIOR_GAMES = 6    # D/ST scoring is much noisier week to week, so lean on last season more
DEF_PRIOR_GAMES = 4    # same, for a defense's points allowed
EPA_PRIOR_PLAYS = 150  # same, for a defense's EPA allowed (in plays)


# ---------------------------------------------------------------- helpers
def _reg_plays(pbp: pd.DataFrame | None) -> pd.DataFrame:
    if pbp is None or pbp.empty:
        return pd.DataFrame()
    p = pbp[(pbp.season_type == "REG") & pbp.play_type.isin(["pass", "run"])]
    return p[p.two_point_attempt != 1]


def _count(personnel: str, codes: tuple) -> int:
    total = 0
    for part in str(personnel).split(","):
        bits = part.strip().split(" ")
        if len(bits) == 2 and bits[1] in codes and bits[0].isdigit():
            total += int(bits[0])
    return total


def offense_grouping(personnel: str) -> str | None:
    """'1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR' -> '11' (RBs then TEs)."""
    if not personnel or personnel != personnel or "QB" not in personnel:
        return None
    return f"{_count(personnel, ('RB', 'FB'))}{_count(personnel, ('TE',))}"


def defense_grouping(personnel: str) -> str | None:
    if not personnel or personnel != personnel:
        return None
    dbs = _count(personnel, ("CB", "FS", "SS", "S", "SAF", "DB"))
    return {4: "Base (4 DB)", 5: "Nickel (5 DB)"}.get(dbs, "Dime+ (6+ DB)" if dbs >= 6 else "Heavy (≤3 DB)")


def _pct(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if len(s) else np.nan


# ---------------------------------------------------------------- players
def player_info(roster: pd.DataFrame | None, roster_prev: pd.DataFrame | None) -> pd.DataFrame:
    frames = [r for r in (roster, roster_prev) if r is not None]
    r = pd.concat(frames)  # current season first, so drop_duplicates keeps current team
    r = r[r.position.isin(POSITIONS) & r.gsis_id.notna()]
    r = r.drop_duplicates("gsis_id")[["gsis_id", "full_name", "team", "position", "status", "headshot_url"]]
    return r.rename(columns={"gsis_id": "player_id", "full_name": "name", "headshot_url": "headshot"}).set_index("player_id")


def fantasy_by_game(pbp: pd.DataFrame | None, ppr: float) -> pd.DataFrame:
    """Per player per game fantasy points (standard scoring + configurable PPR) and usage."""
    p = _reg_plays(pbp)
    if p.empty:
        return pd.DataFrame()
    keys = ["season", "game_id", "week", "posteam", "defteam"]
    f = lambda s: s.fillna(0)

    passing = p[p.passer_player_id.notna()].assign(
        player_id=lambda d: d.passer_player_id,
        pts=lambda d: f(d.passing_yards) * 0.04 + 4 * f(d.pass_touchdown) - 2 * f(d.interception),
    )
    rushing = p[p.rusher_player_id.notna()].assign(
        player_id=lambda d: d.rusher_player_id,
        pts=lambda d: f(d.rushing_yards) * 0.1 + 6 * ((d.rush_touchdown == 1) & (d.td_player_id == d.rusher_player_id)),
        carries=1,
        rz_opps=lambda d: (d.yardline_100 <= 20).astype(int),
    )
    receiving = p[p.receiver_player_id.notna()].assign(
        player_id=lambda d: d.receiver_player_id,
        pts=lambda d: ppr * f(d.complete_pass) + 0.1 * f(d.receiving_yards) + 6 * f(d.pass_touchdown),
        targets=1,
        receptions=lambda d: f(d.complete_pass),
        air_yards=lambda d: f(d.air_yards),
        rz_opps=lambda d: (d.yardline_100 <= 20).astype(int),
    )
    fumbles = p[p.fumbled_1_player_id.notna() & (p.fumble_lost == 1)].assign(
        player_id=lambda d: d.fumbled_1_player_id, pts=-2.0
    )
    # Kickers: FG 0-39 = 3, 40-49 = 4, 50+ = 5, XP = 1, any miss = -1.
    k = pbp[(pbp.season_type == "REG") & pbp.kicker_player_id.notna()
            & ((pbp.field_goal_attempt == 1) | (pbp.extra_point_attempt == 1))]
    fg_made = k.field_goal_result == "made"
    fg_pts = np.select([k.kick_distance < 40, k.kick_distance < 50], [3, 4], 5)
    kicking = k.assign(
        player_id=k.kicker_player_id,
        pts=np.where(k.field_goal_attempt == 1, np.where(fg_made, fg_pts, -1),
                     np.where(k.extra_point_result == "good", 1, -1)),
    )
    stats = ["pts", "carries", "targets", "receptions", "air_yards", "rz_opps"]
    out = pd.concat([d.reindex(columns=keys + ["player_id"] + stats)
                     for d in (passing, rushing, receiving, fumbles, kicking)])
    out[stats] = out[stats].fillna(0)
    out = out.groupby(keys + ["player_id"], as_index=False)[stats].sum()
    return out.rename(columns={"posteam": "team", "defteam": "opp"})


def dst_by_game(pbp: pd.DataFrame | None, games: pd.DataFrame) -> pd.DataFrame:
    """Team defense/special teams fantasy points per game (standard scoring)."""
    if pbp is None or pbp.empty:
        return pd.DataFrame()
    p = pbp[pbp.season_type == "REG"]
    count = lambda mask, team_col: p[mask].groupby(["game_id", team_col]).size()
    events = pd.DataFrame({
        "sacks": count(p.sack == 1, "defteam"),
        "ints": count(p.interception == 1, "defteam"),
        "fum_rec": count(p.fumble_lost == 1, "fumble_recovery_1_team"),
        # Kickoff return TDs belong to posteam (the receiving team); every other non-offense TD to td_team.
        "tds": count((p.touchdown == 1) & ((p.play_type == "kickoff") | (p.td_team != p.posteam)), "td_team"),
        "safeties": count(p.safety == 1, "defteam"),
        "blocks": count((p.field_goal_result == "blocked") | (p.punt_blocked == 1)
                        | (p.extra_point_result == "blocked"), "defteam"),
    }).fillna(0)
    events.index.names = ["game_id", "team"]

    g = games[games.game_id.isin(p.game_id.unique()) & games.home_score.notna()]
    rows = pd.concat([
        pd.DataFrame({"season": g.season, "game_id": g.game_id, "week": g.week, "team": g.home_team,
                      "opp": g.away_team, "pts_allowed": g.away_score}),
        pd.DataFrame({"season": g.season, "game_id": g.game_id, "week": g.week, "team": g.away_team,
                      "opp": g.home_team, "pts_allowed": g.home_score}),
    ])
    rows = rows.join(events, on=["game_id", "team"]).fillna(0)
    tier = rows.pts_allowed.map(lambda pa: next(v for cap, v in PA_TIERS if pa <= cap))
    rows["pts"] = (rows.sacks + 2 * rows.ints + 2 * rows.fum_rec + 6 * rows.tds
                   + 2 * rows.safeties + 2 * rows.blocks + tier)
    rows["player_id"] = "DST_" + rows.team
    for c in ["carries", "targets", "receptions", "air_yards", "rz_opps"]:
        rows[c] = 0
    return rows.drop(columns=["sacks", "ints", "fum_rec", "tds", "safeties", "blocks", "pts_allowed"])


# ---------------------------------------------------------------- team tendencies
FTN_STATS = ["n_blitzers", "n_pass_rushers", "n_defense_box", "is_motion", "is_play_action", "read_thrown",
             "is_catchable_ball", "is_contested_ball", "is_drop", "is_screen_pass", "is_rpo", "qb_location",
             "n_offense_backfield"]
QB_LOCATION = {"U": "Under center", "S": "Shotgun", "P": "Pistol"}


def _join_ftn(pbp: pd.DataFrame, ftn: pd.DataFrame | None) -> pd.DataFrame:
    if ftn is None or ftn.empty:
        return pbp.assign(**{c: np.nan for c in FTN_STATS})
    f = ftn[["nflverse_game_id", "nflverse_play_id"] + FTN_STATS] \
        .rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    f = f.assign(qb_location=f.qb_location.where(f.qb_location.isin(list(QB_LOCATION))))  # "0" = not charted
    return pbp.merge(f, on=["game_id", "play_id"], how="left")


def target_quality_by_game(pbp: pd.DataFrame | None, ftn: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.Series]:
    """FTN-charted targets per receiver per game (first reads, catchable, contested, drops), plus each team's
    first-read throws per game — the denominator for first-read share."""
    p = _join_ftn(_reg_plays(pbp), ftn)
    p = p[p.read_thrown.notna()] if not p.empty else p
    if p.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    first = p.read_thrown.eq("1")
    t = p[p.receiver_player_id.notna()].assign(
        first_read=first, catchable=p.is_catchable_ball.eq(True), contested=p.is_contested_ball.eq(True),
        drop=p.is_drop.eq(True))
    per = t.groupby(["receiver_player_id", "posteam", "game_id"]).agg(
        ftn_tgts=("first_read", "size"), first_reads=("first_read", "sum"), catchable=("catchable", "sum"),
        contested=("contested", "sum"), drops=("drop", "sum")).reset_index()
    team_first = p[first & (p.pass_attempt == 1)].groupby(["posteam", "game_id"]).size()
    return per.rename(columns={"receiver_player_id": "player_id", "posteam": "team"}), team_first


def _join_part(pbp: pd.DataFrame, part: pd.DataFrame | None) -> pd.DataFrame:
    if part is None or part.empty or pbp.empty:
        return pd.DataFrame()
    cols = ["nflverse_game_id", "play_id", "offense_personnel", "defense_personnel",
            "offense_formation", "defense_man_zone_type", "defense_coverage_type", "was_pressure"]
    pa = part[cols].rename(columns={"nflverse_game_id": "game_id"})
    d = pbp.merge(pa, on=["game_id", "play_id"], how="inner")
    d["off_group"] = d.offense_personnel.map(offense_grouping)
    d["def_group"] = d.defense_personnel.map(defense_grouping)
    d["is_man"] = d.defense_man_zone_type.map({"MAN_COVERAGE": 1.0, "ZONE_COVERAGE": 0.0})
    return d


def personnel_season(data: dict) -> tuple[int, pd.DataFrame]:
    """Plays joined with personnel/coverage. Current season if published, otherwise prior."""
    cur = _join_part(_reg_plays(data["pbp"]), data["part"])
    if not cur.empty:
        return data["season"], cur
    return data["season"] - 1, _join_part(_reg_plays(data["pbp_prev"]), data["part_prev"])


def defense_profile(data: dict, fp_all: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """One row per defense: scheme tendencies + what it allows. Blends prior season for stability."""
    season = data["season"]
    cur = _join_ftn(_reg_plays(data["pbp"]), data["ftn"])
    prev = _reg_plays(data["pbp_prev"])
    rows = {}

    def epa_blend(team, mask_fn):
        c = cur[(cur.defteam == team) & mask_fn(cur)].epa.dropna() if not cur.empty else pd.Series(dtype=float)
        pv = prev[(prev.defteam == team) & mask_fn(prev)].epa.dropna() if not prev.empty else pd.Series(dtype=float)
        prior = pv.mean() if len(pv) else 0.0
        return (c.sum() + prior * EPA_PRIOR_PLAYS) / (len(c) + EPA_PRIOR_PLAYS)

    is_pass = lambda d: d.qb_dropback == 1
    is_rush = lambda d: (d.rush_attempt == 1) & (d.qb_scramble != 1)
    teams = sorted(set(cur.defteam.dropna()) | set(prev.defteam.dropna()))
    for t in teams:
        d = cur[cur.defteam == t]
        db = d[d.qb_dropback == 1]
        deep = d[(d.pass_attempt == 1) & (d.air_yards >= 20)]
        runs = d[is_rush(d)]
        rows[t] = {
            "games": d.game_id.nunique(),
            "pass_epa_allowed": epa_blend(t, is_pass),
            "rush_epa_allowed": epa_blend(t, is_rush),
            "blitz_rate": _pct(db.n_blitzers.gt(0).where(db.n_blitzers.notna())),
            "avg_pass_rushers": db.n_pass_rushers.mean(),
            "avg_box": d.n_defense_box.mean(),
            "sack_rate": _pct(db.sack),
            "deep_ypa_allowed": deep.yards_gained.mean() if len(deep) else np.nan,
            "explosive_run_rate": _pct(runs.yards_gained.ge(10)) if len(runs) else np.nan,
            "opp_pass_rate": _pct(d[d.down.isin([1, 2]) & d.wp.between(0.2, 0.8)].qb_dropback),
        }
    prof = pd.DataFrame.from_dict(rows, orient="index")

    # Coverage & sub-package tendencies from participation data.
    part_season, part = personnel_season(data)
    if not part.empty:
        g = part.groupby("defteam")
        prof["man_rate"] = g.is_man.mean()
        prof["pressure_rate"] = part[part.qb_dropback == 1].groupby("defteam").was_pressure.mean()
        prof["top_coverage"] = part.dropna(subset=["defense_coverage_type"]).groupby("defteam") \
            .defense_coverage_type.agg(lambda s: s.value_counts().index[0].replace("_", " ").title())
        for grp in ["Base (4 DB)", "Nickel (5 DB)", "Dime+ (6+ DB)"]:
            prof[grp] = g.def_group.apply(lambda s, grp=grp: (s == grp).mean())
    prof.attrs["personnel_season"] = part_season

    # Fantasy points allowed per game by position, blended with prior season.
    fp = fp_all.join(info.position, on="player_id")
    games = pd.concat([_reg_plays(data["pbp"]), prev]).groupby(["season", "defteam"]).game_id.nunique()
    allowed = fp.groupby(["season", "opp", "position"]).pts.sum()
    for pos in POSITIONS:
        vals = {}
        for t in prof.index:
            g_c = games.get((season, t), 0)
            pts_c = allowed.get((season, t, pos), 0.0)
            g_p = games.get((season - 1, t), 0)
            prior = allowed.get((season - 1, t, pos), 0.0) / g_p if g_p else np.nan
            if np.isnan(prior):
                vals[t] = pts_c / g_c if g_c else np.nan
            else:
                vals[t] = (pts_c + prior * DEF_PRIOR_GAMES) / (g_c + DEF_PRIOR_GAMES)
        prof[f"{pos}_pts_allowed"] = pd.Series(vals)
        prof[f"{pos}_rank"] = prof[f"{pos}_pts_allowed"].rank(ascending=False).astype("Int64")  # 1 = most generous
    return prof


def offense_profile(data: dict) -> pd.DataFrame:
    cur = _join_ftn(_reg_plays(data["pbp"]), data["ftn"])
    rows = {}
    for t, d in cur.groupby("posteam"):
        neutral = d[d.down.isin([1, 2]) & d.wp.between(0.2, 0.8)]
        rows[t] = {
            "plays_per_game": len(d) / max(d.game_id.nunique(), 1),
            "neutral_pass_rate": _pct(neutral.qb_dropback),
            "shotgun_rate": _pct(d.shotgun),
            "no_huddle_rate": _pct(d.no_huddle),
            "motion_rate": _pct(d.is_motion),
            "play_action_rate": _pct(d[d.qb_dropback == 1].is_play_action),
            "epa_per_play": d.epa.mean(),
            "sack_rate_allowed": _pct(d[d.qb_dropback == 1].sack),
            "giveaways_per_game": (d.interception.sum() + d.fumble_lost.sum()) / max(d.game_id.nunique(), 1),
            # FTN formation charting (current season only).
            "under_center_rate": _pct(d.qb_location.eq("U").where(d.qb_location.notna())),
            "pistol_rate": _pct(d.qb_location.eq("P").where(d.qb_location.notna())),
            "empty_rate": _pct(d.n_offense_backfield.eq(0).where(d.n_offense_backfield.notna())),
            "two_back_rate": _pct(d.n_offense_backfield.ge(2).where(d.n_offense_backfield.notna())),
            "screen_rate": _pct(d[d.pass_attempt == 1].is_screen_pass),
            "rpo_rate": _pct(d.is_rpo),
        }
    prof = pd.DataFrame.from_dict(rows, orient="index")
    _, part = personnel_season(data)
    if not part.empty:
        mix = part.groupby("posteam").off_group.value_counts(normalize=True).unstack(fill_value=0)
        for grp in ["11", "12", "21", "13"]:
            if grp in mix:
                prof[f"{grp} personnel"] = mix[grp]
    return prof


def formation_mix(data: dict, team: str, side: str) -> pd.DataFrame:
    """Personnel & formation breakdown for one team, with success rate by grouping."""
    _, part = personnel_season(data)
    if part.empty:
        return pd.DataFrame()
    key = "posteam" if side == "offense" else "defteam"
    d = part[part[key] == team]
    grp = "off_group" if side == "offense" else "def_group"
    out = d.groupby(grp).agg(plays=("epa", "size"), pass_rate=("qb_dropback", "mean"),
                             epa_per_play=("epa", "mean"), success_rate=("success", "mean"))
    out["share"] = out.plays / out.plays.sum()
    return out.sort_values("plays", ascending=False)


def alignment_mix(data: dict, team: str, side: str) -> pd.DataFrame:
    """Current-season FTN formation tendencies: QB alignment and backfield count, with results for each look.
    For a defense, the same split of the looks it has faced and what it allowed."""
    d = _join_ftn(_reg_plays(data["pbp"]), data["ftn"])
    d = d[d["posteam" if side == "offense" else "defteam"] == team]
    backs = d.n_offense_backfield.map(lambda n: np.nan if pd.isna(n) else
                                      "Empty" if n == 0 else "1 back" if n == 1 else "2+ backs")
    parts = []
    for look, order in ((d.qb_location.map(QB_LOCATION), list(QB_LOCATION.values())),
                        (backs, ["Empty", "1 back", "2+ backs"])):
        g = d.assign(look=look).dropna(subset=["look"]).groupby("look").agg(
            plays=("epa", "size"), pass_rate=("qb_dropback", "mean"), epa_per_play=("epa", "mean"),
            success_rate=("success", "mean"))
        g["share"] = g.plays / g.plays.sum()
        parts.append(g.reindex([o for o in order if o in g.index]))
    out = pd.concat(parts)
    out.index.name = None
    return out


def receiver_coverage_splits(data: dict) -> pd.DataFrame:
    """Per receiver: yards/target and fantasy pts/target vs man and vs zone."""
    _, part = personnel_season(data)
    if part.empty:
        return pd.DataFrame()
    t = part[part.receiver_player_id.notna() & part.is_man.notna()]
    t = t.assign(cov=np.where(t.is_man == 1, "man", "zone"), yds=t.receiving_yards.fillna(0))
    s = t.groupby(["receiver_player_id", "cov"]).agg(tgts=("yds", "size"), ypt=("yds", "mean")).unstack()
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    return s.fillna(0)


# ---------------------------------------------------------------- schedule & projections
def week_schedule(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    g = games[(games.season == season) & (games.week == week) & (games.game_type == "REG")]
    rows = []
    for _, r in g.iterrows():
        total, spread = r.get("total_line"), r.get("spread_line")  # spread > 0 means home favored
        has = pd.notna(total) and pd.notna(spread)
        home_it = (total / 2 + spread / 2) if has else np.nan
        away_it = (total / 2 - spread / 2) if has else np.nan
        roof = r.get("roof")
        rows.append({"team": r.home_team, "opp": r.away_team, "home": True, "implied_total": home_it,
                     "gameday": r.gameday, "roof": roof})
        rows.append({"team": r.away_team, "opp": r.home_team, "home": False, "implied_total": away_it,
                     "gameday": r.gameday, "roof": roof})
    return pd.DataFrame(rows).set_index("team") if rows else pd.DataFrame()


def default_week(games: pd.DataFrame, season: int) -> int:
    g = games[(games.season == season) & (games.game_type == "REG")]
    unplayed = g[g.home_score.isna()].groupby("week").size() / g.groupby("week").size()
    upcoming = unplayed[unplayed > 0.5]
    return int(upcoming.index.min()) if len(upcoming) else int(g.week.max())


def project(data: dict, ppr: float, week: int) -> tuple[pd.DataFrame, dict]:
    season = data["season"]
    games = data["games"]
    teams = sorted(set(games[games.season == season].home_team))
    dst_info = pd.DataFrame({"name": [f"{t} D/ST" for t in teams], "team": teams, "position": "DST",
                             "status": "ACT"}, index=pd.Index([f"DST_{t}" for t in teams], name="player_id"))
    info = pd.concat([player_info(data["roster"], data["roster_prev"]), dst_info])
    fp_cur = pd.concat([fantasy_by_game(data["pbp"], ppr), dst_by_game(data["pbp"], games)])
    fp_prev = pd.concat([fantasy_by_game(data["pbp_prev"], ppr), dst_by_game(data["pbp_prev"], games)])
    fp_all = pd.concat([fp_cur, fp_prev])
    defense = defense_profile(data, fp_all, info)
    offense = offense_profile(data)
    sched = week_schedule(data["games"], season, week)
    lg_implied = sched.implied_total.mean() if len(sched) else np.nan
    lg_pass_epa, lg_rush_epa = defense.pass_epa_allowed.mean(), defense.rush_epa_allowed.mean()

    cur = fp_cur.groupby("player_id").agg(g=("pts", "size"), pts=("pts", "sum"), targets=("targets", "sum"),
                                          carries=("carries", "sum"), air=("air_yards", "sum"),
                                          rz=("rz_opps", "sum"), last_team=("team", "last"))
    last3 = fp_cur.sort_values("week").groupby("player_id").tail(3).groupby("player_id").pts.mean()
    prev = fp_prev.groupby("player_id").agg(g_prev=("pts", "size"), ppg_prev=("pts", "mean"))
    team_tgts = fp_cur.groupby(["team", "game_id"]).targets.sum()
    team_car = fp_cur.groupby(["team", "game_id"]).carries.sum()
    played = fp_cur.groupby("player_id").apply(lambda d: list(zip(d.team, d.game_id)))

    df = info.join(cur, how="left").join(prev, how="left").join(last3.rename("last3"))
    df = df[(df.g.fillna(0) > 0) | (df.g_prev.fillna(0) >= 4)]
    df = df[df.status.fillna("ACT").isin(["ACT", "RES"]) | (df.g.fillna(0) > 0)]
    df[["g", "pts", "targets", "carries", "air", "rz", "g_prev"]] = df[["g", "pts", "targets", "carries", "air", "rz", "g_prev"]].fillna(0)

    def share(pid, series, own):
        games = played.get(pid, [])
        tot = sum(series.get(k, 0) for k in games)
        return own / tot if tot else np.nan

    df["tgt_share"] = [share(pid, team_tgts, r.targets) for pid, r in df.iterrows()]
    df["carry_share"] = [share(pid, team_car, r.carries) for pid, r in df.iterrows()]
    df.loc[df.position.isin(["K", "DST"]), ["tgt_share", "carry_share"]] = np.nan

    # Target quality from this season's FTN charting.
    tq, team_first = target_quality_by_game(data["pbp"], data["ftn"])
    q_cols = ["ftn_tgts", "first_reads", "catchable", "contested", "drops"]
    q = tq.groupby("player_id")[q_cols].sum() if len(tq) else pd.DataFrame(columns=q_cols)
    df = df.join(q)
    charted = df.ftn_tgts.where(df.ftn_tgts > 0)
    df["first_read_share"] = [share(pid, team_first, r.first_reads) if r.ftn_tgts > 0 else np.nan
                              for pid, r in df.iterrows()]
    df["catchable_rate"] = df.catchable / charted
    df["contested_rate"] = df.contested / charted
    df.loc[~df.position.isin(["RB", "WR", "TE"]), ["first_read_share", "catchable_rate", "contested_rate", "drops"]] = np.nan
    df["adot"] = np.where(df.targets > 0, df.air / df.targets.replace(0, np.nan), np.nan)
    df["ppg"] = np.where(df.g > 0, df.pts / df.g.replace(0, np.nan), np.nan)
    has_prev = df.ppg_prev.notna()
    prior_g = np.where(df.position == "DST", DST_PRIOR_GAMES, PRIOR_GAMES)
    df["base"] = np.where(has_prev, (df.pts + df.ppg_prev.fillna(0) * prior_g) / (df.g + prior_g), df.ppg)

    team = df.team.fillna(df.last_team)
    df["team"] = team
    df["opp"] = team.map(sched.opp) if len(sched) else np.nan
    df["implied_total"] = team.map(sched.implied_total) if len(sched) else np.nan

    def mults(r):
        if pd.isna(r.opp) or r.opp not in defense.index:
            return pd.Series({"dvp_mult": np.nan, "scheme_mult": np.nan, "env_mult": np.nan})
        d = defense.loc[r.opp]
        lg = defense[f"{r.position}_pts_allowed"].mean()
        dvp = np.clip(d[f"{r.position}_pts_allowed"] / lg, 0.75, 1.30) if lg else 1.0
        env = (r.implied_total / lg_implied) ** 0.6 if pd.notna(r.implied_total) and lg_implied else 1.0
        if r.position == "DST":
            # A D/ST wants a bad opposing offense and a low opposing implied total.
            o_epa = offense.epa_per_play.get(r.opp, np.nan)
            scheme = 1 + np.clip(-(o_epa - offense.epa_per_play.mean()) * 0.5, -0.08, 0.08) if pd.notna(o_epa) else 1.0
            opp_it = sched.implied_total.get(r.opp, np.nan)
            env = np.clip((lg_implied / opp_it) ** 0.5, 0.85, 1.2) if pd.notna(opp_it) and opp_it else 1.0
            dvp = np.clip(dvp, 0.85, 1.15)
            return pd.Series({"dvp_mult": dvp, "scheme_mult": scheme, "env_mult": env})
        pass_diff, rush_diff = d.pass_epa_allowed - lg_pass_epa, d.rush_epa_allowed - lg_rush_epa
        diff = 0.7 * rush_diff + 0.3 * pass_diff if r.position == "RB" else pass_diff
        scheme = 1.0 if r.position == "K" else 1 + np.clip(diff * 0.5, -0.08, 0.08)
        return pd.Series({"dvp_mult": dvp, "scheme_mult": scheme, "env_mult": env})

    df = df.join(df.apply(mults, axis=1))
    df["proj"] = df.base * df.dvp_mult * df.scheme_mult * df.env_mult
    df.loc[df.opp.isna(), "proj"] = 0.0  # bye week
    df = df.sort_values("proj", ascending=False)
    ctx = {"defense": defense, "offense": offense, "schedule": sched, "fp_cur": fp_cur, "fp_all": fp_all,
           "coverage_splits": receiver_coverage_splits(data), "info": info,
           "ftn_through": int(fp_cur[fp_cur.game_id.isin(tq.game_id)].week.max()) if len(tq) else None}
    return df, ctx


def matchup_notes(r: pd.Series, ctx: dict) -> list[str]:
    """Plain-English reasons behind a player's matchup."""
    notes = []
    if pd.isna(r.opp):
        return ["On bye this week."]
    d, dfn = ctx["defense"].loc[r.opp], ctx["defense"]
    rank = d[f"{r.position}_rank"]
    allowed = d[f"{r.position}_pts_allowed"]
    sched, off = ctx["schedule"], ctx["offense"]
    if r.position == "DST":
        notes.append(f"{r.opp}'s offense gives up {allowed:.1f} pts/gm to opposing D/STs (#{rank} most of {len(dfn)}).")
        opp_it = sched.implied_total.get(r.opp, np.nan)
        if pd.notna(opp_it):
            notes.append(f"{r.opp} implied team total: {opp_it:.1f} (league avg {sched.implied_total.mean():.1f}; lower is better).")
        if r.opp in off.index:
            o = off.loc[r.opp]
            notes.append(f"{r.opp} offense: sacked on {o.sack_rate_allowed:.0%} of dropbacks (league "
                         f"{off.sack_rate_allowed.mean():.0%}), {o.giveaways_per_game:.1f} giveaways/gm "
                         f"(league {off.giveaways_per_game.mean():.1f}).")
        return notes
    notes.append(f"{r.opp} allows {allowed:.1f} pts/gm to {r.position}s (#{rank} most of {len(dfn)}).")
    if pd.notna(r.implied_total):
        notes.append(f"{r.team} implied team total: {r.implied_total:.1f}.")
    if r.position == "K":
        roof = sched.roof.get(r.team) if "roof" in sched else None
        if isinstance(roof, str):
            notes.append("Indoor game (dome/closed roof) — no wind or weather risk." if roof in ("dome", "closed")
                         else "Outdoor game — check the wind forecast before kickoff.")
        return notes
    if r.position in ("QB", "WR", "TE") and pd.notna(d.get("blitz_rate")):
        notes.append(f"{r.opp} blitzes on {d.blitz_rate:.0%} of dropbacks (league {dfn.blitz_rate.mean():.0%}).")
    if r.position in ("WR", "TE") and pd.notna(d.get("man_rate")):
        s = ctx["coverage_splits"]
        line = f"{r.opp} plays man {d.man_rate:.0%} of snaps (league {dfn.man_rate.mean():.0%})"
        if r.name in s.index and s.loc[r.name].get("tgts_man", 0) >= 10 and s.loc[r.name].get("tgts_zone", 0) >= 10:
            sp = s.loc[r.name]
            better = "man" if sp.ypt_man > sp.ypt_zone else "zone"
            line += f"; {r['name']} averages {sp.ypt_man:.1f} yds/tgt vs man, {sp.ypt_zone:.1f} vs zone (better vs {better})"
        notes.append(line + ".")
    if r.position in ("WR", "TE") and pd.notna(r.get("first_read_share")) and r.first_read_share > 0.25:
        notes.append(f"Primary read: {r.first_read_share:.0%} of {r.team}'s first-read throws have gone to "
                     f"{r['name']} this season ({r.catchable_rate:.0%} of targets catchable, "
                     f"{int(r.drops)} drop{'s' * (int(r.drops) != 1)}) — volume by design, not leftovers.")
    if r.position in ("WR", "TE") and pd.notna(r.adot) and r.adot >= 12 and pd.notna(d.deep_ypa_allowed):
        notes.append(f"Deep threat (aDOT {r.adot:.1f}); {r.opp} allows {d.deep_ypa_allowed:.1f} yds per deep attempt "
                     f"(league {dfn.deep_ypa_allowed.mean():.1f}).")
    if r.position == "RB":
        if pd.notna(d.avg_box):
            notes.append(f"{r.opp} averages {d.avg_box:.1f} in the box (league {dfn.avg_box.mean():.1f}).")
        if pd.notna(d.explosive_run_rate):
            notes.append(f"{r.opp} allows 10+ yd runs on {d.explosive_run_rate:.0%} of carries "
                         f"(league {dfn.explosive_run_rate.mean():.0%}).")
    if pd.notna(d.get("Nickel (5 DB)")) and r.position in ("WR", "TE"):
        notes.append(f"{r.opp} is in nickel {d['Nickel (5 DB)']:.0%} / dime {d['Dime+ (6+ DB)']:.0%} of snaps.")
    return notes


def optimize_lineup(players: pd.DataFrame, slots: dict) -> pd.DataFrame:
    """Fill fixed position slots with the top projections, then FLEX (RB/WR/TE) and SUPERFLEX (QB/RB/WR/TE).
    Greedy is optimal here because each fixed slot only accepts one position."""
    pool = players.sort_values("proj", ascending=False).copy()
    picks = []
    for pos in POSITIONS:
        for i in range(slots.get(pos, 0)):
            cand = pool[pool.position == pos]
            if len(cand):
                picks.append((f"{pos}{i + 1}" if slots[pos] > 1 else pos, cand.index[0]))
                pool = pool.drop(cand.index[0])
    for slot, allowed in (("FLEX", ["RB", "WR", "TE"]), ("SUPERFLEX", SKILL)):
        for i in range(slots.get(slot, 0)):
            cand = pool[pool.position.isin(allowed)]
            if len(cand):
                picks.append((slot, cand.index[0]))
                pool = pool.drop(cand.index[0])
    lineup = players.loc[[p for _, p in picks]].copy()
    lineup.insert(0, "slot", [s for s, _ in picks])
    return lineup
