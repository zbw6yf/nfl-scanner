import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timezone
import nflreadpy as nfl

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NFL Opportunity Scanner",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 NFL Betting Opportunity Scanner")
st.caption("Personal research tool – Past performance does not guarantee future results. Bet responsibly.")

# -----------------------------
# SIDEBAR – SETTINGS
# -----------------------------
st.sidebar.header("Settings")

api_key = st.sidebar.text_input(
    "The Odds API Key (optional but recommended)",
    type="password",
    help="Get a free key at https://the-odds-api.com"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**How to get a free API key:**")
st.sidebar.markdown("1. Go to [the-odds-api.com](https://the-odds-api.com)")
st.sidebar.markdown("2. Sign up → copy your key")
st.sidebar.markdown("3. Paste it above")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
@st.cache_data(ttl=3600)  # cache for 1 hour
def load_schedules(season=None):
    try:
        if season is None:
            season = nfl.get_current_season()
        sched = nfl.load_schedules(seasons=season)
        return sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    except Exception as e:
        st.error(f"Could not load schedules: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_team_stats(seasons=None):
    try:
        if seasons is None:
            seasons = [nfl.get_current_season() - 1, nfl.get_current_season()]
        stats = nfl.load_team_stats(seasons=seasons)
        return stats.to_pandas() if hasattr(stats, "to_pandas") else stats
    except Exception as e:
        st.warning(f"Team stats limited: {e}")
        return pd.DataFrame()

def fetch_nfl_odds(api_key: str):
    """Fetch current NFL odds from The Odds API"""
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
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Odds API error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        st.error(f"Failed to fetch odds: {e}")
        return None

def american_to_implied_prob(odds):
    """Convert American odds to implied probability"""
    if odds is None or pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

# -----------------------------
# MAIN APP
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Today's Opportunities", "📅 Schedule & Odds", "ℹ️ How it works"])

with tab1:
    st.subheader("Ranked Opportunities (This Week)")

    # Load data
    with st.spinner("Loading NFL data..."):
        schedules = load_schedules()
        team_stats = load_team_stats()
        odds_data = fetch_nfl_odds(api_key) if api_key else None

    # Build opportunities list
    opportunities = []

    if odds_data:
        for game in odds_data:
            home = game.get("home_team")
            away = game.get("away_team")
            commence = game.get("commence_time", "")[:16].replace("T", " ")

            # Get best available lines (simple average across books for now)
            spreads = []
            totals = []
            moneylines_home = []
            moneylines_away = []

            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market["key"] == "spreads":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == home:
                                spreads.append(outcome.get("point"))
                    elif market["key"] == "totals":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == "Over":
                                totals.append(outcome.get("point"))
                    elif market["key"] == "h2h":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == home:
                                moneylines_home.append(outcome.get("price"))
                            elif outcome["name"] == away:
                                moneylines_away.append(outcome.get("price"))

            avg_spread = np.mean(spreads) if spreads else None
            avg_total = np.mean(totals) if totals else None
            avg_ml_home = np.mean(moneylines_home) if moneylines_home else None
            avg_ml_away = np.mean(moneylines_away) if moneylines_away else None

            # --- Simple rule-based + statistical signals ---
            signals = []
            score = 0

            # Rule 1: Home underdog (historically often valuable)
            if avg_spread is not None and avg_spread > 0:
                signals.append("Home underdog")
                score += 1.5

            # Rule 2: Large spread (possible overreaction)
            if avg_spread is not None and abs(avg_spread) >= 7:
                signals.append(f"Large spread ({avg_spread:+.1f})")
                score += 0.8

            # Rule 3: High total (potential shootout or public bias)
            if avg_total is not None and avg_total >= 48:
                signals.append(f"High total ({avg_total:.1f})")
                score += 0.7

            # Placeholder for future EPA / form signals
            # (We can expand this once we have more historical edge data)

            if signals:
                opportunities.append({
                    "Game": f"{away} @ {home}",
                    "Kickoff (UTC)": commence,
                    "Spread": f"{avg_spread:+.1f}" if avg_spread is not None else "—",
                    "Total": f"{avg_total:.1f}" if avg_total is not None else "—",
                    "ML Home": int(avg_ml_home) if avg_ml_home else "—",
                    "ML Away": int(avg_ml_away) if avg_ml_away else "—",
                    "Signals": " • ".join(signals),
                    "Score": round(score, 1)
                })

    if opportunities:
        df_opp = pd.DataFrame(opportunities)
        df_opp = df_opp.sort_values("Score", ascending=False).reset_index(drop=True)
        st.dataframe(df_opp, use_container_width=True, hide_index=True)

        st.info("Higher Score = more signals currently firing. This is a starting point — we will improve the algorithms together.")
    else:
        if not api_key:
            st.warning("👉 Add your free The Odds API key in the sidebar to see live opportunities.")
        else:
            st.info("No strong signals found right now, or odds are still loading. Try refreshing later.")

with tab2:
    st.subheader("Upcoming Games & Current Odds")

    if odds_data:
        rows = []
        for game in odds_data:
            home = game["home_team"]
            away = game["away_team"]
            commence = game.get("commence_time", "")[:16].replace("T", " ")

            # Best moneyline / spread from first book for display
            best_spread = "—"
            best_total = "—"
            for book in game.get("bookmakers", [])[:1]:
                for market in book.get("markets", []):
                    if market["key"] == "spreads":
                        for o in market["outcomes"]:
                            if o["name"] == home:
                                best_spread = f"{o.get('point', 0):+.1f}"
                    if market["key"] == "totals":
                        for o in market["outcomes"]:
                            if o["name"] == "Over":
                                best_total = f"{o.get('point', 0):.1f}"

            rows.append({
                "Away": away,
                "Home": home,
                "Kickoff (UTC)": commence,
                "Spread (Home)": best_spread,
                "Total": best_total
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Enter an Odds API key in the sidebar to load live lines.")

    st.markdown("---")
    st.subheader("Season Schedule (nflverse)")
    if not schedules.empty:
        # Show next few weeks
        upcoming = schedules[schedules["gameday"] >= datetime.now().strftime("%Y-%m-%d")].head(20)
        cols_to_show = [c for c in ["gameday", "weekday", "away_team", "home_team", "gametime"] if c in upcoming.columns]
        st.dataframe(upcoming[cols_to_show], use_container_width=True, hide_index=True)
    else:
        st.write("Schedule data not available yet.")

with tab3:
    st.markdown("""
    ### How this scanner currently works

    **Data sources**
    - Schedules & team stats → free nflverse data (`nflreadpy`)
    - Live odds → The Odds API (free tier available)

    **Current algorithm styles (v1)**
    1. **Rule-based**: Home underdogs, large spreads, high totals
    2. **Statistical**: Simple averages across sportsbooks
    3. **Ready for expansion**: We can add EPA differentials, rest advantages, weather, public betting %, machine learning models, etc.

    **Next improvements we can make together**
    - Better team strength ratings (EPA, DVOA-style)
    - Historical backtesting of each signal
    - Player props
    - Custom rules you define
    - Alerts / email notifications
    - Multi-user version later

    ---
    **Disclaimer**: This is a research and learning tool only.  
    Sports betting involves risk of loss. Never bet more than you can afford to lose.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Built for personal research • Expandable")
