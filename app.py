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
st.caption("Pass/Rush EPA splits + Recent Form weighting + Simple ML. Research tool only.")

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
    try:
        if seasons is None:
            current = nfl.get_current_season()
            seasons = [current - 1
