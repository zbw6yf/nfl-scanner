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
    page_title="NFL Opportunity Scanner – Props Algorithm",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 NFL Betting Opportunity Scanner")
st.caption("EPA + Rules + Rest + Backtest + Player Props Algorithm. Research tool only. Bet responsibly.")

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
st.sidebar.info("Player Props usually need a paid plan on The Odds API. Free plans often return limited or no props.")

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

@st.cache_data(ttl=3600 * 4)
def load_recent_player_stats():
    """Load weekly player stats for recent seasons so we can calculate averages."""
    try:
        current = nfl.get_current_season()
        seasons = [current - 1, current]
        stats = nfl.load_player_stats(seasons=seasons, summary_level="week")
        if hasattr(stats, "to_pandas"):
            stats = stats.to_pandas()
        return stats
    except Exception as e:
        st.warning(f"Could not load player stats: {e}")
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
    if not api_key or not event_
