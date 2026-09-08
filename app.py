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
    page_title="NFL Opportunity Scanner – Full + Props",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 NFL Betting Opportunity Scanner")
st.caption("EPA + Rules + Rest + Backtest + Player Props. Research tool only.")

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
st.sidebar.info("Player Props usually require a higher paid plan on The Odds API.")

# -----------------------------
# DATA FUNCTIONS
# -----------------------------
@st.cache_data(ttl=3600 * 6)
def get_team_epa(seasons=None):
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
        off = pbp.groupby("posteam")["epa"].agg(off_epa="mean").reset_index().rename(columns={"posteam": "team"})
        deff = pbp.groupby("defteam")["epa"].agg(def_epa="mean").reset_index().rename(columns={"defteam": "team"})
        team_epa = off.merge(deff, on="team", how="outer")
        return team_epa.set_index("team")
    except Exception:
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
        "oddsFormat": "american"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def fetch_player_props(api_key: str, event_id: str):
    """Fetch player props for one specific game"""
    if not api_key or not event_id:
        return None

    # Common NFL player prop markets
    markets = ",".join([
        "player_pass_yds",
        "player_pass_tds",
        "player_rush_yds",
        "player_reception_yds",
        "player_receptions",
        "player_anytime_td",
        "player_pass_completions"
    ])

    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american"
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        else:
            return {"error": f"Status {r.status_code}", "message": r.text[:300]}
    except Exception as e:
        return {"error": str(e)}

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Opportunities",
    "📅 Games & Odds",
    "🎯 Player Props",
    "📊 Backtest",
    "🚀 Deploy"
])

# ========== TAB 1: OPPORTUNITIES ==========
with tab1:
    st.subheader("Ranked Game Opportunities")
    with st.spinner("Loading data..."):
        team_epa = get_team_epa()
        schedules = load_schedules()
        odds_data = fetch_nfl_odds(api_key) if api_key else None

    opportunities = []
    if odds_data and not team_epa.empty:
        for game in odds_data:
            home = game.get("home_team")
            away = game.get("away_team")
            commence = game.get("commence_time", "")[:16].replace("T", " ")
            game_date = commence[:10] if commence else datetime.now().strftime("%Y-%m-%d")

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

            home_off = team_epa.loc[home, "off_epa"] if home in team_epa.index else 0
            home_def = team_epa.loc[home, "def_epa"] if home in team_epa.index else 0
            away_off = team_epa.loc[away, "off_epa"] if away in team_epa.index else 0
            away_def = team_epa.loc[away, "def_epa"] if away in team_epa.index else 0
            epa_edge = (home_off - away_def) - (away_off - home_def)

            home_rest = get_rest_days(schedules, home, game_date)
            away_rest = get_rest_days(schedules, away, game_date)
            rest_diff = home_rest - away_rest

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
                signals.append(f"Home rest +{rest_diff}d")
                score += 1.1
            elif rest_diff <= -3:
                signals.append(f"Away rest {rest_diff}d")
                score += 1.0

            if signals:
                opportunities.append({
                    "Game": f"{away} @ {home}",
                    "Kickoff": commence,
                    "Spread": f"{avg_spread:+.1f}" if avg_spread is not None else "—",
                    "Total": f"{avg_total:.1f}" if avg_total is not None else "—",
                    "EPA Edge": f"{epa_edge:+.3f}",
                    "Signals": " • ".join(signals),
                    "Score": round(score, 1)
                })

    if opportunities:
        df = pd.DataFrame(opportunities).sort_values("Score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Add your API key or wait for stronger signals.")

# ========== TAB 2: GAMES ==========
with tab2:
    st.subheader("Upcoming Games & Lines")
    if odds_data:
        rows = []
        for g in odds_data:
            home, away = g["home_team"], g["away_team"]
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
            rows.append({"Away": away, "Home": home, "Kickoff": commence, "Spread": spread, "Total": total})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Enter API key to load games.")

# ========== TAB 3: PLAYER PROPS ==========
with tab3:
    st.subheader("Player Props Scanner")
    st.write("Select a game to load available player props (Passing Yards, Rushing Yards, Receptions, TDs, etc.)")

    if not api_key:
        st.warning("Enter your Odds API key in the sidebar first.")
    elif not odds_data:
        st.info("No games available right now.")
    else:
        # Create dropdown of games
        game_options = {
            f"{g['away_team']} @ {g['home_team']}": g["id"]
            for g in odds_data
        }
        selected_game = st.selectbox("Choose a game", options=list(game_options.keys()))

        if st.button("Load Player Props for this game", type="primary"):
            event_id = game_options[selected_game]
            with st.spinner("Fetching player props... (this can take 10–20 seconds)"):
                props_data = fetch_player_props(api_key, event_id)

            if props_data is None:
                st.error("Failed to fetch props.")
            elif "error" in props_data:
                st.error(f"API returned an error: {props_data.get('error')}")
                st.write("Most free/basic plans do not include full NFL player props. You may need to upgrade your Odds API plan.")
            else:
                # Parse props into a nice table
                rows = []
                for book in props_data.get("bookmakers", []):
                    book_name = book.get("title", book.get("key"))
                    for market in book.get("markets", []):
                        market_key = market.get("key", "")
                        for outcome in market.get("outcomes", []):
                            rows.append({
                                "Book": book_name,
                                "Market": market_key.replace("player_", "").replace("_", " ").title(),
                                "Player": outcome.get("description", outcome.get("name", "")),
                                "Side": outcome.get("name"),
                                "Line": outcome.get("point"),
                                "Odds": outcome.get("price")
                            })

                if rows:
                    props_df = pd.DataFrame(rows)
                    st.success(f"Found {len(props_df)} prop lines")
                    st.dataframe(props_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No player props returned for this game on your current API plan.")

# ========== TAB 4: BACKTEST ==========
with tab4:
    st.subheader("Improved Backtest – EPA Edge")
    st.write("Uses real spread lines from nflverse.")

    min_edge = st.slider("Minimum EPA edge", 0.03, 0.20, 0.06, 0.01)
    seasons_back = st.multiselect("Seasons", [2022, 2023, 2024, 2025], default=[2023, 2024, 2025])

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            try:
                hist_sched = load_schedules(seasons=seasons_back)
                hist_epa = get_team_epa(seasons=seasons_back)
                completed = hist_sched[hist_sched["result"].notna() & hist_sched["spread_line"].notna()]

                results = []
                for _, row in completed.iterrows():
                    home, away = row["home_team"], row["away_team"]
                    if home not in hist_epa.index or away not in hist_epa.index:
                        continue
                    epa_edge = (hist_epa.loc[home, "off_epa"] - hist_epa.loc[away, "def_epa"]) - \
                               (hist_epa.loc[away, "off_epa"] - hist_epa.loc[home, "def_epa"])
                    spread = row["spread_line"]
                    result = row["result"]

                    if epa_edge >= min_edge:
                        covered = result > spread
                        side = "Home"
                    elif epa_edge <= -min_edge:
                        covered = result < spread
                        side = "Away"
                    else:
                        continue
                    results.append({"side": side, "covered": covered})

                if results:
                    res_df = pd.DataFrame(results)
                    st.metric("ATS Win Rate", f"{res_df['covered'].mean():.1%}", delta=f"{len(res_df)} games")
                else:
                    st.warning("No qualifying games.")
            except Exception as e:
                st.error(str(e))

# ========== TAB 5: DEPLOY ==========
with tab5:
    st.subheader("Deploy / Update your app")
    st.markdown("""
    After making changes:
    1. Upload the new `app.py` to your GitHub repo
    2. Go to share.streamlit.io → your app → Reboot
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Full version with Player Props")
