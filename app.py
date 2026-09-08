import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import nflreadpy as nfl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NFL Opportunity Scanner – Advanced Models",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 NFL Opportunity Scanner – Advanced Models")
st.caption("Pass/Rush EPA splits + Recent Form weighting + Simple ML. Research tool only. Bet responsibly.")

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
st.sidebar.info("Advanced models use Pass/Rush EPA, recent form, and a simple ML layer.")

# -----------------------------
# HELPERS
# -----------------------------
TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
}

def to_abbr(name):
    return TEAM_NAME_MAP.get(name, name)

@st.cache_data(ttl=3600 * 6)
def get_advanced_team_metrics(seasons=None):
    """
    Returns advanced metrics with:
    - Season-long Pass/Rush EPA
    - Recent form (last ~6 games) Pass/Rush EPA
    """
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

        # Season-long
        off_pass = pbp[pbp["play_type"] == "pass"].groupby("posteam")["epa"].mean().rename("off_pass_epa")
        off_rush = pbp[pbp["play_type"] == "run"].groupby("posteam")["epa"].mean().rename("off_rush_epa")
        def_pass = pbp[pbp["play_type"] == "pass"].groupby("defteam")["epa"].mean().rename("def_pass_epa")
        def_rush = pbp[pbp["play_type"] == "run"].groupby("defteam")["epa"].mean().rename("def_rush_epa")

        season = pd.concat([off_pass, off_rush, def_pass, def_rush], axis=1)

        # Recent form (last 6 weeks of available data)
        if "week" in pbp.columns and "season" in pbp.columns:
            max_week = pbp["week"].max()
            recent_pbp = pbp[pbp["week"] >= max(1, max_week - 5)]
        else:
            recent_pbp = pbp.tail(int(len(pbp) * 0.25))  # fallback

        off_pass_r = recent_pbp[recent_pbp["play_type"] == "pass"].groupby("posteam")["epa"].mean().rename("off_pass_recent")
        off_rush_r = recent_pbp[recent_pbp["play_type"] == "run"].groupby("posteam")["epa"].mean().rename("off_rush_recent")
        def_pass_r = recent_pbp[recent_pbp["play_type"] == "pass"].groupby("defteam")["epa"].mean().rename("def_pass_recent")
        def_rush_r = recent_pbp[recent_pbp["play_type"] == "run"].groupby("defteam")["epa"].mean().rename("def_rush_recent")

        recent = pd.concat([off_pass_r, off_rush_r, def_pass_r, def_rush_r], axis=1)

        metrics = season.join(recent, how="outer").fillna(0)
        return metrics
    except Exception as e:
        st.warning(f"Advanced metrics error: {e}")
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

def get_rest_days(schedules, team, game_date):
    try:
        team_games = schedules[
            ((schedules["home_team"] == team) | (schedules
