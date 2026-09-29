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

Then open http://localhost:8501. The first launch downloads the NFL data (about a minute).

## Data sources

- [nflverse](https://github.com/nflverse) — play-by-play, rosters, schedules, Vegas lines, participation data
- [FTN Data](https://ftndata.com) via nflverse — charting data (blitzes, box counts, motion)
- [Open-Meteo](https://open-meteo.com) — forecasts and historical weather
- ESPN — live scores and win probability; Sleeper, ESPN and Yahoo fantasy APIs for roster import

## Deploying (Streamlit Community Cloud)

1. At [share.streamlit.io](https://share.streamlit.io), pick this repository and `app.py`.
2. Under **Advanced settings**, choose **Python 3.12**.
3. Optional — Yahoo import: register an app at [developer.yahoo.com/apps/create](https://developer.yahoo.com/apps/create)
   (Fantasy Sports: Read) with the site's URL as its Redirect URI, then paste the `[yahoo]` block from
   `.streamlit/secrets.example.toml` into the app's **Secrets** settings with your values.

Each visitor's Yahoo sign-in and ESPN private-league cookies are kept only in their own browser session
(in server memory for that session, never written to disk or shared with other visitors).
The app uses about 350 MB of memory, within the free tier.
