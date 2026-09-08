import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import nflreadpy as nfl

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NFL Opportunity Scanner – Full Version",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 NFL Betting Opportunity Scanner – Full Version")
st.caption("EPA + Rules + Rest/Short-week signals + Improved Backtest. Research tool only. Bet responsibly.")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Settings")
api_key = st.sidebar.text_input(
    "The Odds API Key",
    type="password",
    help="Get a free key at https://the-odds-api.com"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick links**")
st.sidebar.markdown("- [Get free Odds API key](https://the-odds-api.com)")
st.sidebar.markdown("- [Deploy on Streamlit Cloud](https://share.streamlit.io)")

# -----------------------------
# DATA FUNCTIONS
# -----------------------------
@st.cache_data(ttl=3600 * 6)
def get_team_epa(seasons=None):
    """Calculate Offensive & Defensive EPA per play."""
    try:
        if seasons is None:
            current = nfl.get_current_season()
            seasons = [current - 1, current]

        pbp = nfl.load_pbp(seasons=seasons)
        if hasattr(pbp, "to_pandas"):
            pbp = pbp.to_pandas()

        pbp = pbp[
            (pbp["play_type"].isin(["pass", "run"])) &
            (pbp["epa"].notna()) &
            (pbp["posteam"].notna()) &
            (pbp["defteam"].notna())
        ].copy()

        off = (
            pbp.groupby("posteam")["epa"]
            .agg(off_epa="mean", plays="count")
            .reset_index()
            .rename(columns={"posteam": "team"})
        )
        deff = (
            pbp.groupby("defteam")["epa"]
            .agg(def_epa="mean")
            .reset_index()
            .rename(columns={"defteam": "team"})
        )

        team_epa = off.merge(deff, on="team", how="outer")
        team_epa["net_epa"] = team_epa["off_epa"] - team_epa["def_epa"]
        return team_epa.set_index("team")
    except Exception as e:
        st.warning(f"EPA data issue: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_schedules(seasons=None):
    try:
        if seasons is None:
            current = nfl.get_current_season()
            seasons = list(range(current - 3, current + 1))
        sched = nfl.load_schedules(seasons=seasons)
        return sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    except Exception:
        return pd.DataFrame()

def fetch_nfl_odds(api_key: str):
    if not api_key:
        return None
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def get_rest_days(schedules, team, game_date):
    try:
        team_games = schedules[
            ((schedules["home_team"] == team) | (schedules["away_team"] == team)) &
            (schedules["gameday"] < str(game_date))
        ].sort_values("gameday")
        if team_games.empty:
            return 7
        last = team_games.iloc[-1]["gameday"]
        return max((pd.to_datetime(game_date) - pd.to_datetime(last)).days, 0)
    except Exception:
        return 7

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Today's Opportunities",
    "📅 Games & Odds",
    "📊 Improved Backtest",
    "🚀 Deploy Online"
])

# ========== TAB 1: OPPORTUNITIES ==========
with tab1:
    st.subheader("Ranked Opportunities")

    with st.spinner("Loading EPA, schedules and live odds..."):
        team_epa = get_team_epa()
        schedules = load_schedules()
        odds_data = fetch_nfl_odds(api_key) if api_key else None

    opportunities = []

    if odds_data is not None and not team_epa.empty:
        for game in odds_data:
            home = game.get("home_team")
            away = game.get("away_team")
            commence = game.get("commence_time", "")[:16].replace("T", " ")
            game_date = commence[:10] if commence else datetime.now().strftime("%Y-%m-%d")

            # Average lines
            spreads, totals = [], []
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market["key"] == "spreads":
                        for o in market["outcomes"]:
                            if o["name"] == home:
                                spreads.append(o.get("point"))
                    elif market["key"] == "totals":
                        for o in market["outcomes"]:
                            if o["name"] == "Over":
                                totals.append(o.get("point"))

            avg_spread = np.mean(spreads) if spreads else None
            avg_total = np.mean(totals) if totals else None

            # EPA edge
            home_off = team_epa.loc[home, "off_epa"] if home in team_epa.index else 0
            home_def = team_epa.loc[home, "def_epa"] if home in team_epa.index else 0
            away_off = team_epa.loc[away, "off_epa"] if away in team_epa.index else 0
            away_def = team_epa.loc[away, "def_epa"] if away in team_epa.index else 0
            epa_edge = (home_off - away_def) - (away_off - home_def)

            # Rest
            home_rest = get_rest_days(schedules, home, game_date)
            away_rest = get_rest_days(schedules, away, game_date)
            rest_diff = home_rest - away_rest

            # Signals
            signals = []
            score = 0.0

            if epa_edge > 0.08:
                signals.append(f"Home EPA edge (+{epa_edge:.3f})")
                score += 2.2
            elif epa_edge < -0.08:
                signals.append(f"Away EPA edge ({epa_edge:.3f})")
                score += 2.0

            if avg_spread is not None and avg_spread > 1.5:
                signals.append("Home underdog")
                score += 1.3
            if avg_spread is not None and abs(avg_spread) >= 7:
                signals.append(f"Large spread ({avg_spread:+.1f})")
                score += 0.7
            if avg_total is not None and avg_total >= 48.5:
                signals.append(f"High total ({avg_total:.1f})")
                score += 0.6

            if rest_diff >= 3:
                signals.append(f"Home rest advantage (+{rest_diff}d)")
                score += 1.1
            elif rest_diff <= -3:
                signals.append(f"Away rest advantage ({rest_diff}d)")
                score += 1.0
            if home_rest <= 5:
                signals.append("Home short week")
                score += 0.5
            if away_rest <= 5:
                signals.append("Away short week")
                score += 0.5

            if home_off > 0.05 and away_def > 0.02:
                signals.append("Home offense vs weak D")
                score += 1.0
            if away_off > 0.05 and home_def > 0.02:
                signals.append("Away offense vs weak D")
                score += 0.9

            if signals:
                opportunities.append({
                    "Game": f"{away} @ {home}",
                    "Kickoff": commence,
                    "Spread": f"{avg_spread:+.1f}" if avg_spread is not None else "—",
                    "Total": f"{avg_total:.1f}" if avg_total is not None else "—",
                    "EPA Edge": f"{epa_edge:+.3f}",
                    "Rest Diff": f"{rest_diff:+d}",
                    "Signals": " • ".join(signals),
                    "Score": round(score, 1)
                })

    if opportunities:
        df = pd.DataFrame(opportunities).sort_values("Score", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.success("Higher Score = stronger combination of signals.")
    else:
        if not api_key:
            st.warning("👉 Add your free The Odds API key in the sidebar to see live opportunities.")
        else:
            st.info("No strong signals right now.")

# ========== TAB 2: GAMES ==========
with tab2:
    st.subheader("Upcoming Games & Current Lines")
    if odds_data:
        rows = []
        for g in odds_data:
            home = g["home_team"]
            away = g["away_team"]
            commence = g.get("commence_time", "")[:16].replace("T", " ")
            spread = total = "—"
            for book in g.get("bookmakers", [])[:1]:
                for m in book.get("markets", []):
                    if m["key"] == "spreads":
                        for o in m["outcomes"]:
                            if o["name"] == home:
                                spread = f"{o.get('point', 0):+.1f}"
                    if m["key"] == "totals":
                        for o in m["outcomes"]:
                            if o["name"] == "Over":
                                total = f"{o.get('point', 0):.1f}"
            rows.append({
                "Away": away,
                "Home": home,
                "Kickoff": commence,
                "Spread": spread,
                "Total": total
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Enter an Odds API key in the sidebar to load live lines.")

# ========== TAB 3: IMPROVED BACKTEST ==========
with tab3:
    st.subheader("Improved Historical Backtest – EPA Edge vs Actual Spreads")
    st.write("Uses real closing-style `spread_line` from nflverse schedules.")

    col1, col2 = st.columns(2)
    with col1:
        min_edge = st.slider("Minimum EPA edge", 0.03, 0.20, 0.06, 0.01)
    with col2:
        seasons_back = st.multiselect(
            "Seasons",
            options=[2022, 2023, 2024, 2025],
            default=[2023, 2024, 2025]
        )

    if st.button("Run Improved Backtest", type="primary"):
        with st.spinner("Calculating... (20–60 seconds)"):
            try:
                hist_sched = load_schedules(seasons=seasons_back if seasons_back else True)
                hist_epa = get_team_epa(seasons=seasons_back if seasons_back else None)

                completed = hist_sched[
                    (hist_sched["result"].notna()) &
                    (hist_sched["spread_line"].notna())
                ].copy()

                results = []
                for _, row in completed.iterrows():
                    home = row["home_team"]
                    away = row["away_team"]
                    if home not in hist_epa.index or away not in hist_epa.index:
                        continue

                    h_off = hist_epa.loc[home, "off_epa"]
                    h_def = hist_epa.loc[home, "def_epa"]
                    a_off = hist_epa.loc[away, "off_epa"]
                    a_def = hist_epa.loc[away, "def_epa"]
                    epa_edge = (h_off - a_def) - (a_off - h_def)

                    spread = row["spread_line"]
                    result = row["result"]

                    if epa_edge >= min_edge:
                        side = "Home"
                        covered = result > spread
                    elif epa_edge <= -min_edge:
                        side = "Away"
                        covered = result < spread
                    else:
                        continue

                    results.append({
                        "season": row.get("season"),
                        "side": side,
                        "epa_edge": round(epa_edge, 3),
                        "spread": spread,
                        "result": result,
                        "covered": covered
                    })

                if results:
                    res_df = pd.DataFrame(results)
                    win_rate = res_df["covered"].mean()
                    n = len(res_df)

                    st.metric(
                        f"ATS Win Rate (EPA edge ≥ {min_edge})",
                        f"{win_rate:.1%}",
                        delta=f"Sample: {n} games"
                    )

                    st.write("### Breakdown by side")
                    st.dataframe(
                        res_df.groupby("side")["covered"].agg(["count", "mean"]).round(3),
                        use_container_width=True
                    )

                    st.write("### Sample of recent results")
                    st.dataframe(res_df.tail(12), use_container_width=True, hide_index=True)

                    st.info("This is a research backtest using nflverse closing-style lines. Not financial advice.")
                else:
                    st.warning("No games met the criteria. Try lowering the minimum EPA edge.")
            except Exception as e:
                st.error(f"Backtest error: {e}")

# ========== TAB 4: DEPLOY ==========
with tab4:
    st.subheader("Deploy this app online for free")
    st.markdown("""
    ### Step-by-step instructions

    1. Create a free GitHub account at [github.com](https://github.com) (if you don’t have one).

    2. Create a **new repository**
       - Name it `nfl-scanner` (or any name you like)
       - Make it **Public**
       - Click Create repository

    3. Upload these two files to the repository:
       - `app.py` (this full version)
       - `requirements.txt`

    4. Go to [share.streamlit.io](https://share.streamlit.io)
       - Sign in with GitHub
       - Click **Create app**
       - Select your repository and the `app.py` file
       - Click **Deploy**

    5. After 1–3 minutes you will get a live public link  
       (example: `https://yourname-nfl-scanner.streamlit.app`)

    You can now open the scanner from any phone or computer.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Full final version • EPA + Rest + Improved Backtest")
