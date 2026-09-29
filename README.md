# Matchup Lab 🏈

A fantasy football lineup optimizer that projects players from opponent scheme and personnel data, weather,
Vegas lines and live win probabilities.

- **My Lineup** — import your team from Sleeper, ESPN or Yahoo (or add players by hand) and get the best
  starting lineup, with the reasons behind every pick.
- **Rankings** — weekly projections for QB, RB, WR, TE, K and D/ST.
- **Game Breakdown** — each offense vs. the opposing defense: personnel groupings, blitz/box/coverage
  tendencies, fantasy points allowed and win probabilities.
- **Weather** — kickoff forecasts and how teams, positions and players perform in wind, rain/snow, cold and heat.
- **Live scoreboard** — pregame odds and ESPN live win probability, refreshed every 30 seconds during games.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Then open http://localhost:8501. The first launch downloads NFL data and weather history (a minute or two).

## Data sources

- [nflverse](https://github.com/nflverse) — play-by-play, rosters, schedules, Vegas lines, participation data
- [FTN Data](https://ftndata.com) via nflverse — charting data (blitzes, box counts, motion)
- [Open-Meteo](https://open-meteo.com) — forecasts and historical weather
- ESPN — live scores and win probability; Sleeper, ESPN and Yahoo fantasy APIs for roster import
