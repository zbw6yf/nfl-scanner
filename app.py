import streamlit as st
import pandas as pd
import requests
import re
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import warnings
warnings.filterwarnings("ignore")
try:
    import nflreadpy as nfl
except ImportError:
    st.error("nflreadpy is not installed. Run: pip install nflreadpy")
    st.stop()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NFL Opportunity Scanner",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Theme + polish CSS ----
if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "Dark"

TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00",
    "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LAC": "#0080C6", "LA": "#003594", "LV": "#000000", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SF": "#AA0000",
    "SEA": "#002244", "TB": "#D50A0A", "TEN": "#0C2340", "WAS": "#5A1414",
}

CONF_COLORS = {"A": "#22c55e", "B": "#84cc16", "C": "#eab308", "D": "#f97316", "F": "#6b7280"}


def inject_theme_css(theme: str) -> None:
    dark = theme == "Dark"
    bg = "#0e1117" if dark else "#f7f8fa"
    card = "#1a1f2e" if dark else "#ffffff"
    text = "#e8eaed" if dark else "#1a1d26"
    muted = "#9aa0a6" if dark else "#5f6368"
    accent = "#3b82f6"
    border = "#2d3348" if dark else "#e5e7eb"
    st.markdown(
        f"""
<style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
    h1, h2, h3, h4 {{ letter-spacing: -0.02em; }}
    div[data-testid="stMetric"] {{
        background: {card};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }}
    div[data-testid="stMetric"] label {{ color: {muted} !important; }}
    .nsc-hero {{
        background: linear-gradient(135deg, #0b1220 0%, #1e3a5f 55%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        color: #f8fafc;
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .nsc-hero h1 {{
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc !important;
    }}
    .nsc-hero p {{
        margin: 0.35rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.95rem;
    }}
    .nsc-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        background: rgba(255,255,255,0.12);
        color: #e2e8f0;
    }}
    .nsc-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.65rem;
    }}
    .nsc-card-title {{ font-weight: 650; font-size: 1.02rem; margin-bottom: 0.25rem; color: {text}; }}
    .nsc-muted {{ color: {muted}; font-size: 0.85rem; }}
    .nsc-pill {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 4px;
    }}
    .nsc-stamp {{
        color: {muted};
        font-size: 0.8rem;
        margin: 0.15rem 0 0.75rem 0;
    }}
    .nsc-footer {{
        margin-top: 2rem;
        padding-top: 0.75rem;
        border-top: 1px solid {border};
        color: {muted};
        font-size: 0.8rem;
    }}
    [data-testid="stSidebar"] {{
        background: {"#111827" if dark else "#ffffff"};
        border-right: 1px solid {border};
    }}
</style>
        """,
        unsafe_allow_html=True,
    )


def stamp_now(key: str) -> None:
    st.session_state[f"updated_{key}"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp_text(key: str, label: str) -> str:
    val = st.session_state.get(f"updated_{key}")
    return f"{label}: **{val}**" if val else f"{label}: —"


def current_nfl_week() -> Optional[int]:
    """Best-effort current NFL week from today's date."""
    try:
        return estimate_week_from_date(datetime.now().strftime("%Y-%m-%d"))
    except Exception:
        return None


def conf_pill(grade: str) -> str:
    g = (grade or "F").upper()[:1]
    color = CONF_COLORS.get(g, "#6b7280")
    return f'<span class="nsc-pill" style="background:{color}22;color:{color};border:1px solid {color}55">{g}</span>'


inject_theme_css(st.session_state.get("ui_theme", "Dark"))

st.markdown(
    """
<div class="nsc-hero">
  <span class="nsc-badge">Research tool</span>
  <span class="nsc-badge">EPA · ML · Monte Carlo</span>
  <h1>🏈 NFL Opportunity Scanner</h1>
  <p>Model-driven lean board with schedule, weather, injuries, and depth charts — not betting advice.</p>
</div>
    """,
    unsafe_allow_html=True,
)
# -----------------------------
# CONSTANTS
# -----------------------------
TEAM_NAME_TO_ABBR = {
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
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    "Washington Football Team": "WAS", "Oakland Raiders": "LV",
    "San Diego Chargers": "LAC", "St. Louis Rams": "LA",
}
# Reverse map for display names
ABBR_TO_FULL = {v: k for k, v in TEAM_NAME_TO_ABBR.items() if k not in (
    "Washington Football Team", "Oakland Raiders", "San Diego Chargers", "St. Louis Rams"
)}
# Prefer modern names
ABBR_TO_FULL.update({
    "WAS": "Washington Commanders",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams",
})

STADIUM_COORDS = {
    "ARI": (33.5275, -112.2625), "ATL": (33.7554, -84.4010), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0950, -84.5160), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LAC": (33.9535, -118.3392), "LA": (33.9535, -118.3392),
    "LV": (36.0908, -115.1830), "MIA": (25.9580, -80.2389), "MIN": (44.9738, -93.2581),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8128, -74.0742),
    "NYJ": (40.8128, -74.0742), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SF": (37.4033, -121.9694), "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9077, -76.8645),
}
# Time zone offsets from UTC (standard; DST handled roughly via season)
TEAM_TZ = {
    "ARI": -7, "ATL": -5, "BAL": -5, "BUF": -5, "CAR": -5, "CHI": -6,
    "CIN": -5, "CLE": -5, "DAL": -6, "DEN": -7, "DET": -5, "GB": -6,
    "HOU": -6, "IND": -5, "JAX": -5, "KC": -6, "LAC": -8, "LA": -8,
    "LV": -8, "MIA": -5, "MIN": -6, "NE": -5, "NO": -6, "NYG": -5,
    "NYJ": -5, "PHI": -5, "PIT": -5, "SF": -8, "SEA": -8, "TB": -5,
    "TEN": -6, "WAS": -5,
}
# NFL Divisions (stable alignment)
DIVISIONS = {
    "AFC East": {"BUF", "MIA", "NE", "NYJ"},
    "AFC North": {"BAL", "CIN", "CLE", "PIT"},
    "AFC South": {"HOU", "IND", "JAX", "TEN"},
    "AFC West": {"DEN", "KC", "LAC", "LV"},
    "NFC East": {"DAL", "NYG", "PHI", "WAS"},
    "NFC North": {"CHI", "DET", "GB", "MIN"},
    "NFC South": {"ATL", "CAR", "NO", "TB"},
    "NFC West": {"ARI", "LA", "SF", "SEA"},
}
TEAM_TO_DIV = {}
for div, teams in DIVISIONS.items():
    for t in teams:
        TEAM_TO_DIV[t] = div

def to_abbr(name: str) -> Optional[str]:
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if name in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[name]
    if len(name) <= 3 and name.isupper():
        return name
    return None

def full_name(abbr: str) -> str:
    return ABBR_TO_FULL.get(abbr, abbr)


def format_kickoff(commence_raw: str) -> str:
    """
    Convert Odds API commence_time (UTC ISO) to US/Eastern for display.
    Avoids evening games rolling to the next calendar day in UTC.
    """
    if not commence_raw:
        return ""
    try:
        ts = pd.to_datetime(commence_raw, utc=True)
        try:
            from zoneinfo import ZoneInfo
            ts_et = ts.tz_convert(ZoneInfo("America/New_York"))
        except Exception:
            ts_et = ts.tz_convert(None) - pd.Timedelta(hours=4)
            return ts_et.strftime("%Y-%m-%d %H:%M ET")
        return ts_et.strftime("%Y-%m-%d %H:%M ET")
    except Exception:
        return (commence_raw[:16].replace("T", " ") if len(commence_raw) >= 16 else commence_raw)


def format_schedule_kickoff(gameday: str, gametime: Optional[str]) -> str:
    """Build display kickoff from schedule gameday + gametime (local ET style)."""
    if not gameday:
        return ""
    gd = str(gameday)[:10]
    gt = (str(gametime).strip() if gametime and str(gametime) not in ("None", "nan") else "")
    if gt:
        # gametime is usually "13:00" or "20:15" in Eastern
        try:
            hh, mm = gt.split(":")[:2]
            return f"{gd} {int(hh):02d}:{mm} ET"
        except Exception:
            return f"{gd} {gt} ET"
    return f"{gd} ET"


def is_divisional(home: str, away: str) -> bool:
    return TEAM_TO_DIV.get(home) == TEAM_TO_DIV.get(away) and home in TEAM_TO_DIV

def timezone_diff(home: str, away: str) -> int:
    """Absolute hours of timezone change for the away team traveling to home."""
    h = TEAM_TZ.get(home, -5)
    a = TEAM_TZ.get(away, -5)
    return abs(h - a)

def travel_direction(home: str, away: str) -> str:
    """Rough direction of travel for away team: Eastbound, Westbound, or None."""
    h = TEAM_TZ.get(home, -5)
    a = TEAM_TZ.get(away, -5)
    diff = h - a  # positive = away is traveling west (to earlier TZ)
    if abs(diff) < 1:
        return "None"
    return "Westbound" if diff > 0 else "Eastbound"
# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Settings")
theme_choice = st.sidebar.radio(
    "Theme",
    options=["Dark", "Light"],
    index=0 if st.session_state.get("ui_theme", "Dark") == "Dark" else 1,
    horizontal=True,
    key="theme_radio",
)
if theme_choice != st.session_state.get("ui_theme"):
    st.session_state["ui_theme"] = theme_choice
    inject_theme_css(theme_choice)
    st.rerun()

api_key = st.sidebar.text_input("The Odds API Key", type="password")
n_simulations = st.sidebar.slider("Monte Carlo simulations", 2000, 15000, 8000, 1000)
form_window = st.sidebar.slider("Recent form window (games)", 4, 8, 6, 1)
if st.sidebar.button("Clear all caches"):
    st.cache_data.clear()
    st.cache_resource.clear()
    for k in list(st.session_state.keys()):
        if "weather" in k.lower() or k.startswith("updated_"):
            del st.session_state[k]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Data freshness**")
st.sidebar.caption(stamp_text("odds", "Odds"))
st.sidebar.caption(stamp_text("schedule", "Schedule"))
st.sidebar.caption(stamp_text("weather", "Weather"))
st.sidebar.caption(stamp_text("injuries", "Injuries"))
st.sidebar.caption(stamp_text("depth", "Depth charts"))
st.sidebar.caption("Weather is unique per stadium + kickoff.")
# -----------------------------
# DATA FUNCTIONS
# -----------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_team_epa(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    try:
        if seasons is None:
            current = int(nfl.get_current_season())
            seasons = [current - 1, current]
        pbp = nfl.load_pbp(seasons=seasons)
        if hasattr(pbp, "to_pandas"):
            pbp = pbp.to_pandas()
        if pbp is None or pbp.empty:
            return pd.DataFrame()
        pbp = pbp[
            (pbp["play_type"].isin(["pass", "run"])) &
            (pbp["epa"].notna()) &
            (pbp["posteam"].notna()) &
            (pbp["defteam"].notna())
        ].copy()
        if pbp.empty:
            return pd.DataFrame()
        off = pbp.groupby("posteam")["epa"].mean().reset_index().rename(
            columns={"posteam": "team", "epa": "off_epa"}
        )
        deff = pbp.groupby("defteam")["epa"].mean().reset_index().rename(
            columns={"defteam": "team", "epa": "def_epa"}
        )
        return off.merge(deff, on="team", how="outer").set_index("team")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_team_pace(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """Plays per game (offense + defense snaps approx via play counts)."""
    try:
        if seasons is None:
            current = int(nfl.get_current_season())
            seasons = [current - 1, current]
        pbp = nfl.load_pbp(seasons=seasons)
        if hasattr(pbp, "to_pandas"):
            pbp = pbp.to_pandas()
        if pbp is None or pbp.empty:
            return pd.DataFrame()
        plays = pbp[
            (pbp["play_type"].isin(["pass", "run"])) &
            (pbp["posteam"].notna())
        ].copy()
        if plays.empty:
            return pd.DataFrame()
        g = plays.groupby(["game_id", "posteam"]).size().reset_index(name="off_plays")
        pace = g.groupby("posteam")["off_plays"].mean().reset_index()
        pace.columns = ["team", "plays_per_game"]
        return pace.set_index("team")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_recent_form(seasons: Optional[List[int]] = None, n_games: int = 6) -> Dict[str, Dict]:
    """
    Last N completed games **in the current season only**.
    Prior seasons are ignored so Week 1 form is neutral (n=0).
    Returns dict[team] = {"form_epa": float, "form_margin": float, "n": int}
    """
    try:
        try:
            current = int(nfl.get_current_season())
        except Exception:
            current = datetime.now().year if datetime.now().month >= 3 else datetime.now().year - 1
        cal_year = datetime.now().year if datetime.now().month >= 3 else datetime.now().year - 1
        current = max(current, cal_year)
        # Current season only — never pull prior years for form
        seasons = [current]
        sched = nfl.load_schedules(seasons=seasons)
        if hasattr(sched, "to_pandas"):
            sched = sched.to_pandas()
        if sched is None or sched.empty:
            return {}
        completed = sched[
            sched["result"].notna() &
            sched["home_score"].notna() &
            sched["away_score"].notna()
        ].copy()
        if completed.empty:
            return {}
        completed["gameday"] = pd.to_datetime(completed["gameday"])
        completed = completed.sort_values("gameday")
        epa_df = get_team_epa(seasons)
        form = {}
        all_teams = set(completed["home_team"].unique()) | set(completed["away_team"].unique())
        for team in all_teams:
            mask = (completed["home_team"] == team) | (completed["away_team"] == team)
            team_games = completed.loc[mask].tail(n_games)
            if team_games.empty:
                continue
            margins = []
            epas = []
            for _, row in team_games.iterrows():
                if row["home_team"] == team:
                    margin = float(row["home_score"]) - float(row["away_score"])
                else:
                    margin = float(row["away_score"]) - float(row["home_score"])
                margins.append(margin)
                if not epa_df.empty and team in epa_df.index:
                    opp = row["away_team"] if row["home_team"] == team else row["home_team"]
                    if opp in epa_df.index:
                        team_off = float(epa_df.loc[team, "off_epa"])
                        opp_def = float(epa_df.loc[opp, "def_epa"])
                        epas.append(team_off - opp_def)
            form[team] = {
                "form_margin": float(np.mean(margins)) if margins else 0.0,
                "form_epa": float(np.mean(epas)) if epas else 0.0,
                "n": len(margins)
            }
        return form
    except Exception:
        return {}

def _to_pandas(obj) -> pd.DataFrame:
    """Convert polars/pandas/other schedule objects to a pandas DataFrame safely."""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.DataFrame):
        return obj
    # polars DataFrame
    if hasattr(obj, "to_pandas"):
        try:
            return obj.to_pandas()
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            # polars: to_dict(as_series=False) -> column-oriented dict
            d = obj.to_dict(as_series=False) if "as_series" in str(getattr(obj.to_dict, "__code__", "")) else None
            if d is None:
                try:
                    d = obj.to_dict(as_series=False)
                except TypeError:
                    d = {c: obj[c].to_list() for c in obj.columns}
            return pd.DataFrame(d)
        except Exception:
            pass
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()


# ESPN team abbr -> our standard abbr
_ESPN_ABBR = {
    "WSH": "WAS", "LAR": "LA", "JAC": "JAX",
}


@st.cache_data(ttl=1800, show_spinner=False)
def load_schedules_from_espn(season: int = None, max_week: int = 18) -> pd.DataFrame:
    """
    Fetch the official NFL schedule from ESPN's public scoreboard API.
    Returns a DataFrame with columns compatible with nflverse schedules
    (season, week, gameday, gametime, home_team, away_team, game_type, roof, ...).
    This is the most reliable source for complete weeks (includes NE@JAX, PHI@CHI, etc.).
    """
    if season is None:
        season = datetime.now().year if datetime.now().month >= 3 else datetime.now().year - 1
    rows = []
    for week in range(1, max_week + 1):
        try:
            url = (
                "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
                f"?seasontype=2&week={week}&dates={season}"
            )
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                continue
            data = r.json()
            events = data.get("events") or []
            if not events:
                # No more scheduled weeks
                if week > 1:
                    break
                continue
            for ev in events:
                try:
                    comps = ev.get("competitions") or []
                    if not comps:
                        continue
                    comp = comps[0]
                    competitors = comp.get("competitors") or []
                    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                    if not home or not away:
                        continue
                    home_abbr = (home.get("team") or {}).get("abbreviation") or ""
                    away_abbr = (away.get("team") or {}).get("abbreviation") or ""
                    home_abbr = _ESPN_ABBR.get(home_abbr, home_abbr)
                    away_abbr = _ESPN_ABBR.get(away_abbr, away_abbr)
                    if not home_abbr or not away_abbr:
                        continue

                    # Date/time in UTC ISO from ESPN
                    date_iso = ev.get("date") or comp.get("date") or ""
                    gameday = ""
                    gametime = ""
                    if date_iso:
                        try:
                            ts = pd.to_datetime(date_iso, utc=True)
                            try:
                                from zoneinfo import ZoneInfo
                                ts_et = ts.tz_convert(ZoneInfo("America/New_York"))
                            except Exception:
                                ts_et = ts.tz_convert(None) - pd.Timedelta(hours=4)
                            gameday = ts_et.strftime("%Y-%m-%d")
                            gametime = ts_et.strftime("%H:%M")
                        except Exception:
                            gameday = date_iso[:10]

                    # Completed?
                    status = ((comp.get("status") or {}).get("type") or {}).get("name") or ""
                    home_score = home.get("score")
                    away_score = away.get("score")
                    result = None
                    if status in ("STATUS_FINAL", "STATUS_FULL_TIME") and home_score is not None and away_score is not None:
                        try:
                            result = float(home_score) - float(away_score)
                        except Exception:
                            result = 0.0

                    venue = (comp.get("venue") or {})
                    indoor = venue.get("indoor")
                    roof = "dome" if indoor else "outdoors"

                    rows.append({
                        "game_id": f"{season}_{week:02d}_{away_abbr}_{home_abbr}",
                        "season": season,
                        "game_type": "REG",
                        "week": week,
                        "gameday": gameday,
                        "gametime": gametime,
                        "away_team": away_abbr,
                        "home_team": home_abbr,
                        "away_score": float(away_score) if away_score not in (None, "") else None,
                        "home_score": float(home_score) if home_score not in (None, "") else None,
                        "result": result,
                        "roof": roof,
                        "spread_line": None,
                        "total_line": None,
                        "espn_id": ev.get("id"),
                    })
                except Exception:
                    continue
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def load_schedules_from_nflverse_release() -> pd.DataFrame:
    """
    Load the full multi-season schedule CSV published by nflverse.
    """
    urls = [
        "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv",
        "https://github.com/nflverse/nfldata/raw/master/data/games.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=25)
            if r.status_code != 200 or not r.text or "game_id" not in r.text[:800]:
                continue
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            if not df.empty and "home_team" in df.columns and "gameday" in df.columns:
                return df
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def load_schedules(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load NFL schedules from multiple sources and merge.

    Priority / merge order:
      1) ESPN scoreboard API (most complete live weeks for current season)
      2) nflverse release CSV
      3) nflreadpy package
    Later sources fill gaps; ESPN rows win on conflicts for current season.
    """
    try:
        if seasons is None:
            try:
                current = int(nfl.get_current_season())
            except Exception:
                current = datetime.now().year if datetime.now().month >= 3 else datetime.now().year - 1
            cal_year = datetime.now().year if datetime.now().month >= 3 else datetime.now().year - 1
            current = max(current, cal_year)
            seasons = list(range(current - 3, current + 2))
        seasons = list(dict.fromkeys(int(s) for s in seasons))
        current_season = max(seasons)

        frames = []

        # 1) ESPN – current season full slate
        try:
            espn = load_schedules_from_espn(season=current_season, max_week=18)
            if not espn.empty:
                frames.append(espn)
        except Exception:
            pass

        # 2) nflverse CSV
        try:
            release = load_schedules_from_nflverse_release()
            if not release.empty:
                if "season" in release.columns:
                    release = release[release["season"].isin(seasons)]
                if not release.empty:
                    frames.append(release)
        except Exception:
            pass

        # 3) nflreadpy package
        try:
            for yr in seasons:
                try:
                    raw = nfl.load_schedules(seasons=[yr])
                    pdf = _to_pandas(raw)
                    if pdf is not None and not pdf.empty:
                        frames.append(pdf)
                except Exception:
                    continue
        except Exception:
            pass

        if not frames:
            return pd.DataFrame()

        # Normalize key columns and merge, preferring earlier frames (ESPN first)
        normalized = []
        for f in frames:
            f = f.copy()
            for col in ("home_team", "away_team"):
                if col in f.columns:
                    f[col] = f[col].astype(str).str.upper().replace({
                        "WSH": "WAS", "WFT": "WAS", "LAR": "LA", "STL": "LA",
                        "JAC": "JAX", "GNB": "GB", "KAN": "KC", "NWE": "NE",
                        "NOR": "NO", "SFO": "SF", "TAM": "TB", "OAK": "LV", "LVR": "LV", "SD": "LAC",
                    })
            if "gameday" in f.columns:
                f["gameday"] = f["gameday"].astype(str).str[:10]
            normalized.append(f)

        sched = pd.concat(normalized, ignore_index=True, sort=False)

        # Prefer ESPN/first occurrence per matchup+date
        if "gameday" in sched.columns and "home_team" in sched.columns and "away_team" in sched.columns:
            sched = sched.drop_duplicates(subset=["gameday", "home_team", "away_team"], keep="first")
        elif "game_id" in sched.columns:
            sched = sched.drop_duplicates(subset=["game_id"], keep="first")

        return sched.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_nfl_odds(api_key: str) -> Tuple[Optional[List], str]:
    if not api_key:
        return None, "No API key"
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american"
            },
            timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            remaining = r.headers.get("x-requests-remaining", "?")
            return data, f"OK – {len(data)} events (remaining: {remaining})"
        return None, f"API {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, str(e)

def fetch_player_props(api_key: str, event_id: str) -> Optional[Dict]:
    if not api_key or not event_id:
        return None
    markets = "player_pass_yds,player_pass_tds,player_rush_yds,player_reception_yds,player_receptions,player_anytime_td,player_pass_completions"
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds",
            params={"apiKey": api_key, "regions": "us", "markets": markets, "oddsFormat": "american"},
            timeout=20
        )
        if r.status_code == 200:
            return r.json()
        return {"error": f"Status {r.status_code}", "message": r.text[:300]}
    except Exception as e:
        return {"error": str(e)}

def get_rest_days(schedules: pd.DataFrame, team: str, game_date: str, season: Optional[int] = None) -> Optional[int]:
    """
    Days since this team's previous game **in the same season**.
    Returns None if the team has not played yet this season (e.g. Week 1)
    so callers treat rest as neutral (no advantage).
    """
    try:
        if schedules is None or schedules.empty or "home_team" not in schedules.columns:
            return None
        s = schedules.copy()
        if season is not None and "season" in s.columns:
            s = s[s["season"] == season]
        # Only completed prior games this season (have a result or date strictly before)
        mask = (
            ((s["home_team"] == team) | (s["away_team"] == team)) &
            (s["gameday"].astype(str).str[:10] < str(game_date)[:10])
        )
        prior = s.loc[mask].sort_values("gameday")
        if prior.empty:
            return None  # no game yet this season → no rest edge
        last = str(prior.iloc[-1]["gameday"])[:10]
        return max(int((pd.to_datetime(str(game_date)[:10]) - pd.to_datetime(last)).days), 0)
    except Exception:
        return None


def rest_differential(schedules: pd.DataFrame, home: str, away: str, game_date: str, week: Optional[int] = None, season: Optional[int] = None) -> int:
    """
    Home rest days minus away rest days.
    Forced to 0 in Week 1 or when either team has no prior game this season.
    """
    if week is not None:
        try:
            if int(week) <= 1:
                return 0
        except Exception:
            pass
    h = get_rest_days(schedules, home, game_date, season=season)
    a = get_rest_days(schedules, away, game_date, season=season)
    if h is None or a is None:
        return 0
    return int(h - a)

def get_roof(schedules: pd.DataFrame, home: str, game_date: str) -> str:
    try:
        if schedules.empty or "roof" not in schedules.columns:
            return "outdoors"
        mask = (
            (schedules["home_team"] == home) &
            (schedules["gameday"].astype(str).str[:10] == str(game_date)[:10])
        )
        rows = schedules.loc[mask]
        if not rows.empty:
            roof = rows.iloc[0]["roof"]
            if pd.notna(roof):
                return str(roof).lower().strip()
        home_rows = schedules[schedules["home_team"] == home].dropna(subset=["roof"])
        if not home_rows.empty:
            return str(home_rows.iloc[-1]["roof"]).lower().strip()
    except Exception:
        pass
    return "outdoors"

# Common schedule abbr variants (nflverse sometimes uses LAR / WSH / etc.)
_TEAM_ALIASES = {
    "LA": {"LA", "LAR", "STL"},
    "LAR": {"LA", "LAR", "STL"},
    "LAC": {"LAC", "SD"},
    "LV": {"LV", "OAK", "LVR"},
    "WAS": {"WAS", "WSH", "WFT"},
    "WSH": {"WAS", "WSH", "WFT"},
    "GB": {"GB", "GNB"},
    "KC": {"KC", "KAN"},
    "NE": {"NE", "NWE"},
    "NO": {"NO", "NOR"},
    "SF": {"SF", "SFO"},
    "TB": {"TB", "TAM"},
    "JAC": {"JAX", "JAC"},
    "JAX": {"JAX", "JAC"},
}

def _expand_team(t: str) -> set:
    return _TEAM_ALIASES.get(t, {t}) | {t}

def estimate_week_from_date(game_date: str) -> Optional[int]:
    """
    Estimate NFL week from calendar date when schedule lookup fails.

    NFL weeks run roughly Thursday → following Wednesday (MNF included).
    Week 1 anchor = first Thursday on/after Sept 4 of the season year.
    Games 1–3 days before that Thursday (Wed openers) still count as Week 1.
    """
    try:
        target = pd.to_datetime(str(game_date)[:10], errors="coerce")
        if pd.isna(target):
            return None
        target = pd.Timestamp(year=target.year, month=target.month, day=target.day)
        year = target.year if target.month >= 3 else target.year - 1

        week1_thu = pd.Timestamp(year=year, month=9, day=4)
        while week1_thu.weekday() != 3:  # Thursday = 3
            week1_thu += pd.Timedelta(days=1)

        if target < week1_thu - pd.Timedelta(days=3):
            if target < week1_thu - pd.Timedelta(days=10):
                return None
            return 1

        days_since_thu = (target - week1_thu).days
        week = days_since_thu // 7 + 1
        if week < 1:
            return 1
        if week > 22:
            return None
        return int(week)
    except Exception:
        return None

def get_week(schedules: pd.DataFrame, home: str, away: str, game_date: str) -> Optional[int]:
    """
    Assign NFL week for an upcoming game.

    Priority:
      1) Current-season schedule match within ±2 days of kickoff (home/away + aliases)
      2) Calendar estimate from kickoff date
      3) If schedule week and estimate disagree by >= 1, prefer the estimate
    """
    est = estimate_week_from_date(game_date)
    try:
        gd = str(game_date)[:10]
        target = pd.to_datetime(gd, errors="coerce")
        if pd.isna(target):
            return est

        try:
            current = int(nfl.get_current_season())
        except Exception:
            current = target.year if target.month >= 8 else target.year - 1

        sched = schedules.copy() if schedules is not None and not getattr(schedules, "empty", True) else pd.DataFrame()
        if not sched.empty and "season" in sched.columns:
            sched_cur = sched[sched["season"] == current]
            if not sched_cur.empty:
                sched = sched_cur

        sched_week = None
        if not sched.empty and "week" in sched.columns and "home_team" in sched.columns:
            home_set = _expand_team(home)
            away_set = _expand_team(away)

            def _week_if_close(rows: pd.DataFrame, max_days: int = 2) -> Optional[int]:
                if rows.empty:
                    return None
                tmp = rows.copy()
                tmp["_gd"] = pd.to_datetime(tmp["gameday"], errors="coerce")
                tmp = tmp.dropna(subset=["_gd"])
                if tmp.empty:
                    return None
                tmp["_diff"] = (tmp["_gd"] - target).abs().dt.days
                tmp = tmp[tmp["_diff"] <= max_days].sort_values("_diff")
                if tmp.empty:
                    return None
                w = tmp.iloc[0]["week"]
                return int(w) if pd.notna(w) else None

            mask = sched["home_team"].isin(home_set) & sched["away_team"].isin(away_set)
            sched_week = _week_if_close(sched.loc[mask], max_days=2)
            if sched_week is None:
                mask_flip = sched["home_team"].isin(away_set) & sched["away_team"].isin(home_set)
                sched_week = _week_if_close(sched.loc[mask_flip], max_days=2)

        if sched_week is not None and est is not None:
            if sched_week == est:
                return sched_week
            return est
        if sched_week is not None:
            return sched_week
        return est
    except Exception:
        return est


def implied_team_totals(spread: float, total: float) -> Tuple[float, float]:
    """
    spread = home team line (negative if home favorite).
    Returns (home_implied, away_implied).
    """
    home_imp = (total - spread) / 2.0
    away_imp = (total + spread) / 2.0
    return home_imp, away_imp


def american_to_implied_prob(american_odds) -> Optional[float]:
    """Convert American odds to implied probability (no-vig raw)."""
    try:
        o = float(american_odds)
    except Exception:
        return None
    if o < 0:
        return (-o) / ((-o) + 100.0)
    if o > 0:
        return 100.0 / (o + 100.0)
    return None


def remove_vig_two_way(p_home: float, p_away: float) -> Tuple[float, float]:
    """Normalize two-way implied probs so they sum to 1."""
    s = p_home + p_away
    if s <= 0:
        return 0.5, 0.5
    return p_home / s, p_away / s


def extract_moneylines(odds_ev: Optional[Dict], home_full: str, away_full: str) -> Tuple[Optional[float], Optional[float]]:
    """Average American moneylines for home/away from an Odds API event."""
    if not odds_ev:
        return None, None
    home_mls, away_mls = [], []
    for book in odds_ev.get("bookmakers", []) or []:
        for market in book.get("markets", []) or []:
            if market.get("key") != "h2h":
                continue
            for o in market.get("outcomes", []) or []:
                name = o.get("name")
                price = o.get("price")
                if price is None:
                    continue
                if name == home_full:
                    home_mls.append(float(price))
                elif name == away_full:
                    away_mls.append(float(price))
    h = float(np.mean(home_mls)) if home_mls else None
    a = float(np.mean(away_mls)) if away_mls else None
    return h, a


def market_home_win_prob(odds_ev: Optional[Dict], home_full: str, away_full: str, spread: Optional[float] = None) -> Optional[float]:
    """
    Fair market probability home wins (or covers if only spread available).
    Prefer moneyline; fall back to rough spread→prob conversion.
    """
    h_ml, a_ml = extract_moneylines(odds_ev, home_full, away_full)
    if h_ml is not None and a_ml is not None:
        ph = american_to_implied_prob(h_ml)
        pa = american_to_implied_prob(a_ml)
        if ph is not None and pa is not None:
            ph, pa = remove_vig_two_way(ph, pa)
            return ph
    # Fallback: approximate cover/win prob from spread (home line)
    if spread is not None:
        # logistic-ish: P(home covers) ≈ 0.5 + spread/25, clamped
        try:
            # Negative spread = home favorite → higher cover prob of the spread line itself isn't win prob
            # For outright win approx: favorite by S points → P(win) ≈ 0.5 + min(0.35, abs(S)*0.03)*sign
            s = float(spread)
            # home spread line: negative means home favored by |s|
            edge = -s  # positive if home favorite
            p = 0.5 + max(-0.35, min(0.35, edge * 0.03))
            return float(p)
        except Exception:
            return None
    return None


def compute_edge(model_prob: float, market_prob: Optional[float]) -> Optional[float]:
    """Model probability edge vs market (percentage points)."""
    if market_prob is None:
        return None
    return (model_prob - market_prob) * 100.0



def confidence_grade(
    rec: str,
    total_score: float,
    ml_home: float,
    mc: dict,
    edge_pct: Optional[float],
    n_signals: int,
    agree: float,
) -> str:
    """
    A/B/C/D/F confidence for the lean (no E).
    A = strongest alignment of score, model, Monte Carlo, and market edge.
    F = no lean or very weak evidence.
    """
    if rec == "No strong lean":
        if total_score >= 4.0 and n_signals >= 3:
            return "D"  # some signals but no formal lean
        return "F"

    # Strength of the lean itself
    if rec in ("Lean Home ATS", "Lean Away ATS"):
        side_prob = mc.get("home_cover_prob", 0.5) if rec == "Lean Home ATS" else (1.0 - mc.get("home_cover_prob", 0.5))
        model_side = ml_home if rec == "Lean Home ATS" else (1.0 - ml_home)
        side_ev = mc.get("home_ev", 0.0) if rec == "Lean Home ATS" else mc.get("away_ev", 0.0)
    else:
        side_prob = mc.get("over_prob", 0.5) if rec == "Lean Over" else mc.get("under_prob", 0.5)
        model_side = side_prob
        side_ev = max(0.0, side_prob - 0.5)

    edge_abs = abs(edge_pct) if edge_pct is not None else 0.0
    points = 0.0
    # Score contribution (0–4)
    points += min(4.0, total_score / 2.5)
    # Probability margin past 50% (0–2)
    points += min(2.0, max(0.0, (side_prob - 0.5) * 10.0))
    # Model agreement (0–1.5)
    if agree > 0:
        points += 1.5
    elif abs(model_side - 0.5) >= 0.05:
        points += 0.75
    # Market edge (0–1.5)
    if edge_abs >= 8:
        points += 1.5
    elif edge_abs >= 4:
        points += 1.0
    elif edge_abs >= 2:
        points += 0.5
    # Signal count (0–1)
    if n_signals >= 5:
        points += 1.0
    elif n_signals >= 3:
        points += 0.5
    # EV quality (0–1)
    if side_ev >= 0.08:
        points += 1.0
    elif side_ev >= 0.04:
        points += 0.5

    # Map points to letter (A/B/C/D/F only — no E)
    if points >= 8.5:
        return "A"
    if points >= 7.0:
        return "B"
    if points >= 5.5:
        return "C"
    if points >= 3.5:
        return "D"
    return "F"



def clv_spread(line_taken: float, closing_line: float, side: str) -> float:
    """
    Closing line value in points for an ATS bet.
    side: 'home' or 'away' — line_taken is the home spread when side=home,
    or the away spread (usually -home_spread) when side=away.
    Positive CLV = you got a better number than close.
    """
    try:
        lt = float(line_taken)
        cl = float(closing_line)
    except Exception:
        return 0.0
    side = (side or "home").lower()
    if side == "home":
        # Took home +3, closed home +1 → CLV = +2 (better)
        return lt - cl
    # Away side: away line is typically -home_line
    # Took away -3 (home was +3), close away -1 (home +1) → better by 2
    return cl - lt


def clv_total(line_taken: float, closing_total: float, side: str) -> float:
    """CLV for totals in points. Over: higher line taken is better. Under: lower is better."""
    try:
        lt = float(line_taken)
        cl = float(closing_total)
    except Exception:
        return 0.0
    side = (side or "over").lower()
    if side == "over":
        return lt - cl  # took 47, closed 45.5 → +1.5
    return cl - lt  # under: took 44, closed 45.5 → +1.5





SLUG_TO_ABBR = {
    "arizona-cardinals": "ARI", "atlanta-falcons": "ATL", "baltimore-ravens": "BAL",
    "buffalo-bills": "BUF", "carolina-panthers": "CAR", "chicago-bears": "CHI",
    "cincinnati-bengals": "CIN", "cleveland-browns": "CLE", "dallas-cowboys": "DAL",
    "denver-broncos": "DEN", "detroit-lions": "DET", "green-bay-packers": "GB",
    "houston-texans": "HOU", "indianapolis-colts": "IND", "jacksonville-jaguars": "JAX",
    "kansas-city-chiefs": "KC", "las-vegas-raiders": "LV", "los-angeles-chargers": "LAC",
    "los-angeles-rams": "LA", "miami-dolphins": "MIA", "minnesota-vikings": "MIN",
    "new-england-patriots": "NE", "new-orleans-saints": "NO", "new-york-giants": "NYG",
    "new-york-jets": "NYJ", "philadelphia-eagles": "PHI", "pittsburgh-steelers": "PIT",
    "san-francisco-49ers": "SF", "seattle-seahawks": "SEA", "tampa-bay-buccaneers": "TB",
    "tennessee-titans": "TEN", "washington-commanders": "WAS",
}

ESPN_TEAM_IDS = {
    "ARI": 22, "ATL": 1, "BAL": 33, "BUF": 2, "CAR": 29, "CHI": 3, "CIN": 4, "CLE": 5,
    "DAL": 6, "DEN": 7, "DET": 8, "GB": 9, "HOU": 34, "IND": 11, "JAX": 30, "KC": 12,
    "LAC": 24, "LA": 14, "LV": 13, "MIA": 15, "MIN": 16, "NE": 17, "NO": 18, "NYG": 19,
    "NYJ": 20, "PHI": 21, "PIT": 23, "SF": 25, "SEA": 26, "TB": 27, "TEN": 10, "WAS": 28,
}

# Ourlads uses LAR for Rams
OURLADS_ABBR = {**{a: a for a in ESPN_TEAM_IDS}, "LA": "LAR", "WAS": "WAS"}




def american_profit(units: float, american_odds: float, won: bool) -> float:
    if not won:
        return -abs(units)
    try:
        o = float(american_odds)
    except Exception:
        o = -110.0
    if o < 0:
        return abs(units) * (100.0 / (-o))
    return abs(units) * (o / 100.0)


BET_LOG_PATH = Path("/home/workdir/artifacts/bet_log.csv")


def _load_bet_log() -> pd.DataFrame:
    cols = [
        "id", "logged_at", "week", "game", "bet_type", "side", "line_taken",
        "odds", "units", "model_prob", "market_prob", "edge_pct",
        "closing_line", "result", "profit_units", "clv", "notes",
    ]
    if "bet_log_df" in st.session_state and isinstance(st.session_state.get("bet_log_df"), pd.DataFrame):
        return st.session_state["bet_log_df"]
    try:
        if BET_LOG_PATH.exists():
            df = pd.read_csv(BET_LOG_PATH)
            st.session_state["bet_log_df"] = df
            return df
    except Exception:
        pass
    df = pd.DataFrame(columns=cols)
    st.session_state["bet_log_df"] = df
    return df


def _save_bet_log(df: pd.DataFrame) -> None:
    st.session_state["bet_log_df"] = df
    try:
        df.to_csv(BET_LOG_PATH, index=False)
    except Exception:
        pass


SIGNAL_HISTORY_PATH = Path("/home/workdir/artifacts/signal_history.csv")


def _load_signal_history() -> pd.DataFrame:
    cols = [
        "id", "logged_at", "week", "game", "home", "away", "kickoff",
        "recommendation", "confidence", "score", "spread", "total",
        "result", "correct", "graded_at",
    ]
    if "signal_history_df" in st.session_state and isinstance(st.session_state.get("signal_history_df"), pd.DataFrame):
        return st.session_state["signal_history_df"]
    try:
        if SIGNAL_HISTORY_PATH.exists():
            df = pd.read_csv(SIGNAL_HISTORY_PATH)
            st.session_state["signal_history_df"] = df
            return df
    except Exception:
        pass
    df = pd.DataFrame(columns=cols)
    st.session_state["signal_history_df"] = df
    return df


def _save_signal_history(df: pd.DataFrame) -> None:
    st.session_state["signal_history_df"] = df
    try:
        df.to_csv(SIGNAL_HISTORY_PATH, index=False)
    except Exception:
        pass


def _upsert_signals_from_opportunities(opps: list) -> None:
    """Add new Game Signals rows to history (skip duplicates by week+game+recommendation)."""
    if not opps:
        return
    hist = _load_signal_history()
    existing = set()
    if not hist.empty:
        for _, r in hist.iterrows():
            existing.add((str(r.get("week")), str(r.get("game")), str(r.get("recommendation"))))
    new_rows = []
    import uuid
    for o in opps:
        game = str(o.get("Game", ""))
        rec = str(o.get("Recommendation", ""))
        week = str(o.get("Week", ""))
        key = (week, game, rec)
        if key in existing:
            continue
        if rec in ("", "—", "None"):
            continue
        new_rows.append({
            "id": str(uuid.uuid4())[:8],
            "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "week": o.get("Week"),
            "game": game,
            "home": o.get("_home"),
            "away": o.get("_away"),
            "kickoff": o.get("Kickoff"),
            "recommendation": rec,
            "confidence": o.get("Confidence"),
            "score": o.get("Score"),
            "spread": o.get("_spread") if o.get("_spread") is not None else o.get("Spread"),
            "total": o.get("_total") if o.get("_total") is not None else o.get("Total"),
            "result": "Pending",
            "correct": None,
            "graded_at": None,
        })
        existing.add(key)
    if new_rows:
        hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)
        _save_signal_history(hist)


def _grade_signal_history(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Grade signals ONLY for fully completed games.

    A game is completed when:
      - kickoff/gameday calendar date is strictly before today
      - home_score and away_score are both present (not null)
      - schedule row matches the SAME matchup on that date (not a prior-year meeting)

    Any previously graded row that fails these checks is reset to Pending.
    """
    hist = _load_signal_history()
    if hist.empty:
        return hist

    today = pd.Timestamp.now().normalize()

    def _parse_day(val) -> Optional[pd.Timestamp]:
        try:
            ts = pd.to_datetime(str(val)[:10], errors="coerce")
            if pd.isna(ts):
                return None
            return pd.Timestamp(ts).normalize()
        except Exception:
            return None

    # ---- Build completed-game index from schedule ----
    completed = pd.DataFrame()
    if schedules is not None and not getattr(schedules, "empty", True):
        s = schedules.copy()
        for col in ("home_team", "away_team"):
            if col in s.columns:
                s[col] = s[col].astype(str).str.upper().str.strip()
        if "gameday" in s.columns:
            s["_gd"] = pd.to_datetime(s["gameday"], errors="coerce")
            s = s[s["_gd"].notna() & (s["_gd"] < today)]
            if "home_score" in s.columns and "away_score" in s.columns:
                s = s[s["home_score"].notna() & s["away_score"].notna()]
                # Drop bogus 0-0 placeholders on missing finals when possible:
                # keep rows that have a non-null result OR any points scored OR explicit final
                if "result" in s.columns:
                    s = s[s["result"].notna() | ((s["home_score"].astype(float) + s["away_score"].astype(float)) > 0)]
            completed = s

    changed = False

    for idx, row in hist.iterrows():
        kick_day = _parse_day(row.get("kickoff"))
        home = str(row.get("home") or "").upper().strip()
        away = str(row.get("away") or "").upper().strip()
        rec = str(row.get("recommendation") or "")
        status = str(row.get("result") or "Pending")

        # Future kickoff → always Pending
        if kick_day is not None and kick_day >= today:
            if status != "Pending":
                hist.at[idx, "result"] = "Pending"
                hist.at[idx, "correct"] = None
                hist.at[idx, "graded_at"] = None
                changed = True
            continue

        # No strong lean never grades as win/loss
        if rec == "No strong lean":
            if status != "N/A":
                hist.at[idx, "result"] = "N/A"
                hist.at[idx, "correct"] = None
                hist.at[idx, "graded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                changed = True
            continue

        if not home or not away or home in ("NAN", "NONE") or away in ("NAN", "NONE"):
            if status not in ("Pending", "N/A"):
                hist.at[idx, "result"] = "Pending"
                hist.at[idx, "correct"] = None
                hist.at[idx, "graded_at"] = None
                changed = True
            continue

        if completed.empty or kick_day is None:
            # Without a kickoff date we refuse to grade (prevents prior-year matchup hits)
            if status not in ("Pending", "N/A"):
                hist.at[idx, "result"] = "Pending"
                hist.at[idx, "correct"] = None
                hist.at[idx, "graded_at"] = None
                changed = True
            continue

        # Strict match: same teams + gameday within 1 day of kickoff date
        mask = (
            (completed["home_team"] == home)
            & (completed["away_team"] == away)
            & (completed["_gd"] >= kick_day - pd.Timedelta(days=1))
            & (completed["_gd"] <= kick_day + pd.Timedelta(days=1))
        )
        matches = completed.loc[mask]
        if matches.empty:
            if status not in ("Pending", "N/A"):
                hist.at[idx, "result"] = "Pending"
                hist.at[idx, "correct"] = None
                hist.at[idx, "graded_at"] = None
                changed = True
            continue

        mrow = matches.sort_values("_gd").iloc[-1]
        try:
            hs = float(mrow["home_score"])
            aws = float(mrow["away_score"])
        except Exception:
            if status not in ("Pending", "N/A"):
                hist.at[idx, "result"] = "Pending"
                hist.at[idx, "correct"] = None
                hist.at[idx, "graded_at"] = None
                changed = True
            continue

        margin = hs - aws
        total_pts = hs + aws

        spread = row.get("spread")
        total_line = row.get("total")
        try:
            if spread is not None and str(spread) not in ("—", "nan", "None", ""):
                spread = float(str(spread).replace("+", ""))
            elif pd.notna(mrow.get("spread_line")):
                spread = float(mrow["spread_line"])
            else:
                spread = None
        except Exception:
            spread = None
        try:
            if total_line is not None and str(total_line) not in ("—", "nan", "None", ""):
                total_line = float(str(total_line).replace("+", ""))
            elif pd.notna(mrow.get("total_line")):
                total_line = float(mrow["total_line"])
            else:
                total_line = None
        except Exception:
            total_line = None

        new_result = None
        new_correct = None
        if rec == "Lean Home ATS" and spread is not None:
            if abs(margin - spread) < 1e-9:
                new_result, new_correct = "Push", None
            else:
                ok = margin > spread
                new_result, new_correct = ("Correct" if ok else "Incorrect"), bool(ok)
        elif rec == "Lean Away ATS" and spread is not None:
            if abs(margin - spread) < 1e-9:
                new_result, new_correct = "Push", None
            else:
                ok = margin < spread
                new_result, new_correct = ("Correct" if ok else "Incorrect"), bool(ok)
        elif rec == "Lean Over" and total_line is not None:
            if abs(total_pts - total_line) < 1e-9:
                new_result, new_correct = "Push", None
            else:
                ok = total_pts > total_line
                new_result, new_correct = ("Correct" if ok else "Incorrect"), bool(ok)
        elif rec == "Lean Under" and total_line is not None:
            if abs(total_pts - total_line) < 1e-9:
                new_result, new_correct = "Push", None
            else:
                ok = total_pts < total_line
                new_result, new_correct = ("Correct" if ok else "Incorrect"), bool(ok)
        else:
            # Cannot grade without lines
            if status not in ("Pending", "N/A"):
                hist.at[idx, "result"] = "Pending"
                hist.at[idx, "correct"] = None
                hist.at[idx, "graded_at"] = None
                changed = True
            continue

        if status != new_result or (hist.at[idx, "correct"] != new_correct):
            hist.at[idx, "result"] = new_result
            hist.at[idx, "correct"] = new_correct
            hist.at[idx, "graded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            changed = True

    if changed:
        _save_signal_history(hist)
    return hist



def build_team_history(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Per-team ATS and Over/Under results for the last ~5 seasons.
    One row per team-game with columns used for filtering/aggregation.
    """
    try:
        try:
            current = int(nfl.get_current_season())
        except Exception:
            current = datetime.now().year if datetime.now().month >= 3 else datetime.now().year - 1
        cal_year = datetime.now().year if datetime.now().month >= 3 else datetime.now().year - 1
        current = max(current, cal_year)
        if seasons is None:
            seasons = list(range(current - 4, current + 1))
        sched = load_schedules(seasons=seasons)
        if sched is None or sched.empty:
            return pd.DataFrame()
        s = sched.copy()
        need = {"home_team", "away_team", "home_score", "away_score"}
        if not need.issubset(set(s.columns)):
            return pd.DataFrame()
        s = s[s["home_score"].notna() & s["away_score"].notna()].copy()
        if "game_type" in s.columns:
            s = s[s["game_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]
        if s.empty:
            return pd.DataFrame()

        rows = []
        for _, g in s.iterrows():
            try:
                hs = float(g["home_score"])
                aws = float(g["away_score"])
            except Exception:
                continue
            margin = hs - aws
            total_pts = hs + aws
            spread = g.get("spread_line")
            total_line = g.get("total_line")
            try:
                spread = float(spread) if pd.notna(spread) else None
            except Exception:
                spread = None
            try:
                total_line = float(total_line) if pd.notna(total_line) else None
            except Exception:
                total_line = None
            season = int(g["season"]) if pd.notna(g.get("season")) else None
            week = g.get("week")
            gameday = str(g.get("gameday", ""))[:10]
            home = str(g["home_team"]).upper()
            away = str(g["away_team"]).upper()

            # Home perspective
            if spread is not None:
                if abs(margin - spread) < 1e-9:
                    home_ats = "Push"
                else:
                    home_ats = "Cover" if margin > spread else "Not Cover"
            else:
                home_ats = None
            if total_line is not None:
                if abs(total_pts - total_line) < 1e-9:
                    ou = "Push"
                else:
                    ou = "Over" if total_pts > total_line else "Under"
            else:
                ou = None

            rows.append({
                "season": season, "week": week, "gameday": gameday,
                "team": home, "opponent": away, "home_away": "Home",
                "spread": spread, "margin": margin, "ats": home_ats,
                "total_line": total_line, "total_pts": total_pts, "ou": ou,
            })
            # Away perspective (away spread is -home spread)
            if spread is not None:
                away_spread = -spread
                away_margin = -margin
                if abs(away_margin - away_spread) < 1e-9:
                    away_ats = "Push"
                else:
                    away_ats = "Cover" if away_margin > away_spread else "Not Cover"
            else:
                away_ats = None
            rows.append({
                "season": season, "week": week, "gameday": gameday,
                "team": away, "opponent": home, "home_away": "Away",
                "spread": -spread if spread is not None else None,
                "margin": -margin, "ats": away_ats,
                "total_line": total_line, "total_pts": total_pts, "ou": ou,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()



def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _http_get(url: str, timeout: int = 30) -> Optional[requests.Response]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        return requests.get(url, headers=headers, timeout=timeout)
    except Exception:
        return None


def _injuries_from_nfl_com() -> pd.DataFrame:
    cols = ["Team", "Team Abbr", "Player", "Position", "Injury", "Practice Status", "Game Status"]
    empty = pd.DataFrame(columns=cols)
    r = _http_get("https://www.nfl.com/injuries/")
    if r is None or r.status_code != 200 or not r.text or len(r.text) < 1000:
        return empty
    html = r.text
    team_order = []
    for m in re.finditer(r'href="(/teams/[a-z0-9-]+/)"', html):
        slug = m.group(1).strip("/").split("/")[-1]
        if slug in SLUG_TO_ABBR and (not team_order or team_order[-1] != slug):
            team_order.append(slug)
    tables = re.findall(r"<table[^>]*>(.*?)</table>", html, flags=re.S | re.I)
    if not tables:
        return empty
    rows = []
    n = min(len(team_order), len(tables)) if team_order else 0
    for i in range(n):
        slug = team_order[i]
        abbr = SLUG_TO_ABBR.get(slug, slug[:3].upper())
        try:
            team_name = full_name(abbr)
        except Exception:
            team_name = abbr
        trs = re.findall(r"<tr[^>]*>(.*?)</tr>", tables[i], flags=re.S | re.I)
        for tr in trs[1:]:
            cells = [_strip_html(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)]
            if len(cells) < 2:
                continue
            player = cells[0]
            if not player or player.lower() == "player":
                continue
            rows.append({
                "Team": team_name,
                "Team Abbr": abbr,
                "Player": player,
                "Position": cells[1] if len(cells) > 1 else "",
                "Injury": (cells[2] if len(cells) > 2 else "") or "—",
                "Practice Status": (cells[3] if len(cells) > 3 else "") or "—",
                "Game Status": (cells[4] if len(cells) > 4 else "") or "—",
            })
    return pd.DataFrame(rows) if rows else empty


def _injuries_from_espn_core() -> pd.DataFrame:
    """ESPN core API — team injury lists + detail pages (slower but reliable)."""
    cols = ["Team", "Team Abbr", "Player", "Position", "Injury", "Practice Status", "Game Status"]
    rows = []
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    for abbr, tid in ESPN_TEAM_IDS.items():
        try:
            url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{tid}/injuries?limit=100"
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200:
                continue
            data = r.json()
            items = data.get("items") or []
            team_name = full_name(abbr)
            # Cap per team to keep load reasonable
            for item in items[:25]:
                ref = (item.get("$ref") or "").replace("http://", "https://")
                if not ref:
                    continue
                try:
                    detail = requests.get(ref, headers=headers, timeout=12).json()
                except Exception:
                    continue
                status = detail.get("status") or "—"
                # Skip pure "Active" noise if no injury designation
                if str(status).lower() == "active":
                    continue
                ath_ref = ((detail.get("athlete") or {}).get("$ref") or "").replace("http://", "https://")
                player = "—"
                pos = ""
                if ath_ref:
                    try:
                        ath = requests.get(ath_ref, headers=headers, timeout=10).json()
                        player = ath.get("displayName") or ath.get("fullName") or "—"
                        p = ath.get("position")
                        if isinstance(p, dict):
                            pos = p.get("abbreviation") or p.get("displayName") or ""
                        elif isinstance(p, str):
                            pos = p
                    except Exception:
                        # Fall back to short comment name guess
                        sc = detail.get("shortComment") or detail.get("longComment") or ""
                        player = sc.split("(")[0].strip()[:40] or "—"
                injury = (detail.get("type") or {}).get("description") if isinstance(detail.get("type"), dict) else ""
                if not injury:
                    # parse from shortComment e.g. "Love (ankle) is..."
                    sc = detail.get("shortComment") or ""
                    m = re.search(r"\(([^)]+)\)", sc)
                    injury = m.group(1) if m else (detail.get("longComment") or "—")[:60]
                rows.append({
                    "Team": team_name,
                    "Team Abbr": abbr,
                    "Player": player,
                    "Position": pos,
                    "Injury": injury or "—",
                    "Practice Status": "—",
                    "Game Status": status,
                })
        except Exception:
            continue
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


@st.cache_data(ttl=900, show_spinner=False)
def load_nfl_injury_report() -> pd.DataFrame:
    """
    Injury report with fallbacks:
      1) NFL.com official HTML report
      2) ESPN core API per-team injuries
    """
    cols = ["Team", "Team Abbr", "Player", "Position", "Injury", "Practice Status", "Game Status"]
    try:
        df = _injuries_from_nfl_com()
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    try:
        df = _injuries_from_espn_core()
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=cols)


def _parse_ourlads_player(cell: str) -> str:
    """'Coleman, Keon 24/2' -> 'Keon Coleman' when possible."""
    cell = _strip_html(cell)
    if not cell:
        return ""
    # Drop draft suffix like 24/2 or U/LAC or CF16
    cell = re.sub(r"\s+\d{2}/\d.*$", "", cell)
    cell = re.sub(r"\s+[A-Z]{1,3}/[A-Z]{2,3}$", "", cell)
    cell = re.sub(r"\s+CF\d+.*$", "", cell)
    cell = re.sub(r"\s+U/[A-Za-z]+$", "", cell)
    if "," in cell:
        parts = [p.strip() for p in cell.split(",", 1)]
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}".strip()
    return cell.strip()


@st.cache_data(ttl=3600, show_spinner=False)
def load_depth_charts() -> pd.DataFrame:
    """
    Load depth charts from Ourlads (https://www.ourlads.com/nfldepthcharts/).
    Returns Team, Team Abbr, Side, Position, Rank, Player
    """
    cols = ["Team", "Team Abbr", "Unit", "Position", "Rank", "Player"]
    rows = []
    for abbr in ESPN_TEAM_IDS.keys():
        ol = OURLADS_ABBR.get(abbr, abbr)
        url = f"https://www.ourlads.com/nfldepthcharts/depthchart/{ol}"
        try:
            r = _http_get(url, timeout=25)
            if r is None or r.status_code != 200 or not r.text:
                continue
            tables = re.findall(r"<table[^>]*>(.*?)</table>", r.text, flags=re.S | re.I)
            try:
                team_name = full_name(abbr)
            except Exception:
                team_name = abbr
            # Heuristic unit labels by table order: Offense, Defense, Special Teams
            unit_names = ["Offense", "Defense", "Special Teams", "Other"]
            for ti, table in enumerate(tables):
                unit = unit_names[ti] if ti < len(unit_names) else "Other"
                trs = re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.S | re.I)
                if not trs:
                    continue
                for tr in trs[1:]:
                    cells = [_strip_html(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)]
                    if len(cells) < 3:
                        continue
                    pos = cells[0]
                    if not pos or pos.lower() in ("pos", "position"):
                        continue
                    # Pattern: Pos, No, Player1, No, Player2, ...
                    rank = 0
                    for i in range(2, len(cells), 2):
                        raw = cells[i] if i < len(cells) else ""
                        player = _parse_ourlads_player(raw)
                        if not player:
                            continue
                        rank += 1
                        rows.append({
                            "Team": team_name,
                            "Team Abbr": abbr,
                            "Unit": unit,
                            "Position": pos,
                            "Rank": rank,
                            "Player": player,
                        })
        except Exception:
            continue
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)



def _normalize_team_abbr(t: str) -> str:
    t = str(t or "").strip().upper()
    aliases = {
        "LAR": "LA", "STL": "LA", "WSH": "WAS", "WFT": "WAS", "JAC": "JAX",
        "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO", "SFO": "SF",
        "TAM": "TB", "OAK": "LV", "LVR": "LV", "SD": "LAC",
    }
    return aliases.get(t, t)


def _extract_odds_lines(odds_ev, home_abbr: str):
    """Return (avg_spread home, avg_total) from an Odds API event."""
    if not odds_ev:
        return None, None
    spreads, totals = [], []
    home_full = odds_ev.get("home_team", full_name(home_abbr))
    for book in odds_ev.get("bookmakers", []) or []:
        for market in book.get("markets", []) or []:
            if market.get("key") == "spreads":
                for o in market.get("outcomes", []) or []:
                    if o.get("name") == home_full and o.get("point") is not None:
                        spreads.append(o.get("point"))
            elif market.get("key") == "totals":
                for o in market.get("outcomes", []) or []:
                    if o.get("name") == "Over" and o.get("point") is not None:
                        totals.append(o.get("point"))
    avg_spread = float(np.mean(spreads)) if spreads else None
    avg_total = float(np.mean(totals)) if totals else None
    return avg_spread, avg_total


def build_upcoming_from_odds(odds_data, schedules: pd.DataFrame) -> List[Dict]:
    """Fallback: build game list purely from Odds API events."""
    games = []
    if not odds_data:
        return games
    for ev in odds_data:
        home_full = ev.get("home_team", "")
        away_full = ev.get("away_team", "")
        home = _normalize_team_abbr(to_abbr(home_full) or "")
        away = _normalize_team_abbr(to_abbr(away_full) or "")
        if not home or not away:
            continue
        commence_raw = ev.get("commence_time") or ""
        kickoff = format_kickoff(commence_raw) if commence_raw else ""
        game_date = commence_raw[:10] if len(commence_raw) >= 10 else ""
        avg_spread, avg_total = _extract_odds_lines(ev, home)
        if avg_total is None:
            avg_total = 45.0
        week = None
        if game_date:
            week = get_week(schedules, home, away, game_date) if (schedules is not None and not getattr(schedules, "empty", True)) else estimate_week_from_date(game_date)
        roof = get_roof(schedules, home, game_date) if (schedules is not None and not getattr(schedules, "empty", True)) else "outdoors"
        games.append({
            "week": week,
            "gameday": game_date,
            "gametime": None,
            "kickoff": kickoff,
            "home": home,
            "away": away,
            "home_full": home_full or full_name(home),
            "away_full": away_full or full_name(away),
            "roof": roof,
            "avg_spread": avg_spread,
            "avg_total": avg_total,
            "odds_event": ev,
            "commence_raw": commence_raw,
            "game_id": ev.get("id"),
        })
    games.sort(key=lambda g: (g.get("gameday") or "", g.get("kickoff") or ""))
    return games





# Embedded FULL 2026 REG schedule Weeks 1-18 (ESPN official)
# Source of truth for matchups / dates / times — never drop games from this list.
EMBEDDED_2026_SCHEDULE = [
    {"week": 1, "gameday": "2026-09-09", "gametime": "20:20", "away": "NE", "home": "SEA", "roof": "outdoors"},
    {"week": 1, "gameday": "2026-09-10", "gametime": "20:35", "away": "SF", "home": "LA", "roof": "outdoors"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "13:00", "away": "TB", "home": "CIN", "roof": "outdoors"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "13:00", "away": "NO", "home": "DET", "roof": "dome"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "13:00", "away": "NYJ", "home": "TEN", "roof": "outdoors"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "13:00", "away": "BAL", "home": "IND", "roof": "dome"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "13:00", "away": "ATL", "home": "PIT", "roof": "outdoors"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "13:00", "away": "CHI", "home": "CAR", "roof": "outdoors"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "13:00", "away": "CLE", "home": "JAX", "roof": "outdoors"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "13:00", "away": "BUF", "home": "HOU", "roof": "dome"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "16:25", "away": "MIA", "home": "LV", "roof": "dome"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "16:25", "away": "GB", "home": "MIN", "roof": "dome"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "16:25", "away": "WAS", "home": "PHI", "roof": "outdoors"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "16:25", "away": "ARI", "home": "LAC", "roof": "outdoors"},
    {"week": 1, "gameday": "2026-09-13", "gametime": "20:20", "away": "DAL", "home": "NYG", "roof": "outdoors"},
    {"week": 1, "gameday": "2026-09-14", "gametime": "20:15", "away": "DEN", "home": "KC", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-17", "gametime": "20:15", "away": "DET", "home": "BUF", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "13:00", "away": "CAR", "home": "ATL", "roof": "dome"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "13:00", "away": "MIN", "home": "CHI", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "13:00", "away": "PHI", "home": "TEN", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "13:00", "away": "PIT", "home": "NE", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "13:00", "away": "GB", "home": "NYJ", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "13:00", "away": "CLE", "home": "TB", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "13:00", "away": "NO", "home": "BAL", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "13:00", "away": "CIN", "home": "HOU", "roof": "dome"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "16:05", "away": "JAX", "home": "DEN", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "16:05", "away": "LV", "home": "LAC", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "16:25", "away": "WAS", "home": "DAL", "roof": "dome"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "16:25", "away": "SEA", "home": "ARI", "roof": "dome"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "16:25", "away": "MIA", "home": "SF", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-20", "gametime": "20:20", "away": "IND", "home": "KC", "roof": "outdoors"},
    {"week": 2, "gameday": "2026-09-21", "gametime": "20:15", "away": "NYG", "home": "LA", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-24", "gametime": "20:15", "away": "ATL", "home": "GB", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "13:00", "away": "LAC", "home": "BUF", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "13:00", "away": "CAR", "home": "CLE", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "13:00", "away": "NYJ", "home": "DET", "roof": "dome"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "13:00", "away": "HOU", "home": "IND", "roof": "dome"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "13:00", "away": "KC", "home": "MIA", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "13:00", "away": "TEN", "home": "NYG", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "13:00", "away": "CIN", "home": "PIT", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "13:00", "away": "SEA", "home": "WAS", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "13:00", "away": "NE", "home": "JAX", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "16:05", "away": "ARI", "home": "SF", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "16:05", "away": "MIN", "home": "TB", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "16:25", "away": "BAL", "home": "DAL", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "16:25", "away": "LV", "home": "NO", "roof": "dome"},
    {"week": 3, "gameday": "2026-09-27", "gametime": "20:20", "away": "LA", "home": "DEN", "roof": "outdoors"},
    {"week": 3, "gameday": "2026-09-28", "gametime": "20:15", "away": "PHI", "home": "CHI", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-01", "gametime": "20:15", "away": "PIT", "home": "CLE", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "09:30", "away": "IND", "home": "WAS", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "13:00", "away": "NE", "home": "BUF", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "13:00", "away": "NYJ", "home": "CHI", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "13:00", "away": "JAX", "home": "CIN", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "13:00", "away": "ARI", "home": "NYG", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "13:00", "away": "LA", "home": "PHI", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "13:00", "away": "GB", "home": "TB", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "13:00", "away": "TEN", "home": "BAL", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "13:00", "away": "DAL", "home": "HOU", "roof": "dome"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "16:05", "away": "MIA", "home": "MIN", "roof": "dome"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "16:25", "away": "KC", "home": "LV", "roof": "dome"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "16:25", "away": "DEN", "home": "SF", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "16:25", "away": "LAC", "home": "SEA", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-04", "gametime": "20:20", "away": "DET", "home": "CAR", "roof": "outdoors"},
    {"week": 4, "gameday": "2026-10-05", "gametime": "20:15", "away": "ATL", "home": "NO", "roof": "dome"},
    {"week": 5, "gameday": "2026-10-08", "gametime": "20:15", "away": "TB", "home": "DAL", "roof": "dome"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "09:30", "away": "PHI", "home": "JAX", "roof": "outdoors"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "13:00", "away": "HOU", "home": "TEN", "roof": "outdoors"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "13:00", "away": "CIN", "home": "MIA", "roof": "outdoors"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "13:00", "away": "LV", "home": "NE", "roof": "outdoors"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "13:00", "away": "MIN", "home": "NO", "roof": "dome"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "13:00", "away": "CLE", "home": "NYJ", "roof": "outdoors"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "13:00", "away": "IND", "home": "PIT", "roof": "outdoors"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "13:00", "away": "NYG", "home": "WAS", "roof": "outdoors"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "16:05", "away": "DEN", "home": "LAC", "roof": "outdoors"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "16:25", "away": "CHI", "home": "GB", "roof": "outdoors"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "16:25", "away": "DET", "home": "ARI", "roof": "dome"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "16:25", "away": "SF", "home": "SEA", "roof": "outdoors"},
    {"week": 5, "gameday": "2026-10-11", "gametime": "20:20", "away": "BAL", "home": "ATL", "roof": "dome"},
    {"week": 5, "gameday": "2026-10-12", "gametime": "20:15", "away": "BUF", "home": "LA", "roof": "outdoors"},
    {"week": 6, "gameday": "2026-10-15", "gametime": "20:15", "away": "SEA", "home": "DEN", "roof": "outdoors"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "09:30", "away": "HOU", "home": "JAX", "roof": "outdoors"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "13:00", "away": "CHI", "home": "ATL", "roof": "dome"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "13:00", "away": "BAL", "home": "CLE", "roof": "outdoors"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "13:00", "away": "TEN", "home": "IND", "roof": "dome"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "13:00", "away": "NYJ", "home": "NE", "roof": "outdoors"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "13:00", "away": "NO", "home": "NYG", "roof": "outdoors"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "13:00", "away": "CAR", "home": "PHI", "roof": "outdoors"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "13:00", "away": "PIT", "home": "TB", "roof": "outdoors"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "16:05", "away": "ARI", "home": "LA", "roof": "outdoors"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "16:25", "away": "LAC", "home": "KC", "roof": "outdoors"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "16:25", "away": "BUF", "home": "LV", "roof": "dome"},
    {"week": 6, "gameday": "2026-10-18", "gametime": "20:20", "away": "DAL", "home": "GB", "roof": "outdoors"},
    {"week": 6, "gameday": "2026-10-19", "gametime": "20:15", "away": "WAS", "home": "SF", "roof": "outdoors"},
    {"week": 7, "gameday": "2026-10-22", "gametime": "20:15", "away": "NE", "home": "CHI", "roof": "outdoors"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "09:30", "away": "PIT", "home": "NO", "roof": "outdoors"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "13:00", "away": "SF", "home": "ATL", "roof": "dome"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "13:00", "away": "CLE", "home": "TEN", "roof": "outdoors"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "13:00", "away": "IND", "home": "MIN", "roof": "dome"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "13:00", "away": "MIA", "home": "NYJ", "roof": "outdoors"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "13:00", "away": "TB", "home": "CAR", "roof": "outdoors"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "13:00", "away": "CIN", "home": "BAL", "roof": "outdoors"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "13:00", "away": "NYG", "home": "HOU", "roof": "dome"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "16:05", "away": "DEN", "home": "ARI", "roof": "dome"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "16:25", "away": "GB", "home": "DET", "roof": "dome"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "16:25", "away": "LA", "home": "LV", "roof": "dome"},
    {"week": 7, "gameday": "2026-10-25", "gametime": "20:20", "away": "KC", "home": "SEA", "roof": "outdoors"},
    {"week": 7, "gameday": "2026-10-26", "gametime": "20:15", "away": "DAL", "home": "PHI", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-10-29", "gametime": "20:15", "away": "CAR", "home": "GB", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "13:00", "away": "BAL", "home": "BUF", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "13:00", "away": "TEN", "home": "CIN", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "13:00", "away": "ARI", "home": "DAL", "roof": "dome"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "13:00", "away": "MIN", "home": "DET", "roof": "dome"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "13:00", "away": "LV", "home": "NYJ", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "13:00", "away": "CLE", "home": "PIT", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "13:00", "away": "ATL", "home": "TB", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "13:00", "away": "IND", "home": "JAX", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "16:05", "away": "LAC", "home": "LA", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "16:25", "away": "KC", "home": "DEN", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "16:25", "away": "NE", "home": "MIA", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-11-01", "gametime": "20:20", "away": "PHI", "home": "WAS", "roof": "outdoors"},
    {"week": 8, "gameday": "2026-11-02", "gametime": "20:15", "away": "CHI", "home": "SEA", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-05", "gametime": "20:15", "away": "JAX", "home": "BAL", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "09:30", "away": "CIN", "home": "ATL", "roof": "dome"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "13:00", "away": "DAL", "home": "IND", "roof": "dome"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "13:00", "away": "NYJ", "home": "KC", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "13:00", "away": "DET", "home": "MIA", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "13:00", "away": "CLE", "home": "NO", "roof": "dome"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "13:00", "away": "NYG", "home": "PHI", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "13:00", "away": "LA", "home": "WAS", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "13:00", "away": "DEN", "home": "CAR", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "16:05", "away": "HOU", "home": "LAC", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "16:05", "away": "LV", "home": "SF", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "16:25", "away": "GB", "home": "NE", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "16:25", "away": "ARI", "home": "SEA", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-08", "gametime": "20:20", "away": "TB", "home": "CHI", "roof": "outdoors"},
    {"week": 9, "gameday": "2026-11-09", "gametime": "20:15", "away": "BUF", "home": "MIN", "roof": "dome"},
    {"week": 10, "gameday": "2026-11-12", "gametime": "20:15", "away": "WAS", "home": "NYG", "roof": "outdoors"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "09:30", "away": "NE", "home": "DET", "roof": "outdoors"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "13:00", "away": "KC", "home": "ATL", "roof": "dome"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "13:00", "away": "HOU", "home": "CLE", "roof": "outdoors"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "13:00", "away": "MIN", "home": "GB", "roof": "outdoors"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "13:00", "away": "JAX", "home": "TEN", "roof": "outdoors"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "13:00", "away": "MIA", "home": "IND", "roof": "dome"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "13:00", "away": "CAR", "home": "NO", "roof": "dome"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "13:00", "away": "BUF", "home": "NYJ", "roof": "outdoors"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "16:05", "away": "SEA", "home": "LV", "roof": "dome"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "16:05", "away": "LA", "home": "ARI", "roof": "dome"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "16:25", "away": "SF", "home": "DAL", "roof": "dome"},
    {"week": 10, "gameday": "2026-11-15", "gametime": "20:20", "away": "PIT", "home": "CIN", "roof": "outdoors"},
    {"week": 10, "gameday": "2026-11-16", "gametime": "20:15", "away": "LAC", "home": "BAL", "roof": "outdoors"},
    {"week": 11, "gameday": "2026-11-19", "gametime": "20:15", "away": "IND", "home": "HOU", "roof": "dome"},
    {"week": 11, "gameday": "2026-11-22", "gametime": "13:00", "away": "MIA", "home": "BUF", "roof": "outdoors"},
    {"week": 11, "gameday": "2026-11-22", "gametime": "13:00", "away": "NO", "home": "CHI", "roof": "outdoors"},
    {"week": 11, "gameday": "2026-11-22", "gametime": "13:00", "away": "TEN", "home": "DAL", "roof": "dome"},
    {"week": 11, "gameday": "2026-11-22", "gametime": "13:00", "away": "TB", "home": "DET", "roof": "dome"},
    {"week": 11, "gameday": "2026-11-22", "gametime": "13:00", "away": "ARI", "home": "KC", "roof": "outdoors"},
    {"week": 11, "gameday": "2026-11-22", "gametime": "13:00", "away": "JAX", "home": "NYG", "roof": "outdoors"},
    {"week": 11, "gameday": "2026-11-22", "gametime": "13:00", "away": "BAL", "home": "CAR", "roof": "outdoors"},
    {"week": 11, "gameday": "2026-11-22", "gametime": "16:05", "away": "NYJ", "home": "LAC", "roof": "outdoors"},
    {"week": 11, "gameday": "2026-11-22", "gametime": "16:25", "away": "LV", "home": "DEN", "roof": "outdoors"},
    {"week": 11, "gameday": "2026-11-22", "gametime": "16:25", "away": "PIT", "home": "PHI", "roof": "outdoors"},
    {"week": 11, "gameday": "2026-11-22", "gametime": "20:20", "away": "MIN", "home": "SF", "roof": "outdoors"},
    {"week": 11, "gameday": "2026-11-23", "gametime": "20:15", "away": "CIN", "home": "WAS", "roof": "outdoors"},
    {"week": 12, "gameday": "2026-11-25", "gametime": "20:00", "away": "GB", "home": "LA", "roof": "outdoors"},
    {"week": 12, "gameday": "2026-11-26", "gametime": "13:00", "away": "CHI", "home": "DET", "roof": "dome"},
    {"week": 12, "gameday": "2026-11-26", "gametime": "16:30", "away": "PHI", "home": "DAL", "roof": "dome"},
    {"week": 12, "gameday": "2026-11-26", "gametime": "20:20", "away": "KC", "home": "BUF", "roof": "outdoors"},
    {"week": 12, "gameday": "2026-11-27", "gametime": "15:00", "away": "DEN", "home": "PIT", "roof": "outdoors"},
    {"week": 12, "gameday": "2026-11-29", "gametime": "13:00", "away": "NO", "home": "CIN", "roof": "outdoors"},
    {"week": 12, "gameday": "2026-11-29", "gametime": "13:00", "away": "LV", "home": "CLE", "roof": "outdoors"},
    {"week": 12, "gameday": "2026-11-29", "gametime": "13:00", "away": "NYG", "home": "IND", "roof": "dome"},
    {"week": 12, "gameday": "2026-11-29", "gametime": "13:00", "away": "NYJ", "home": "MIA", "roof": "outdoors"},
    {"week": 12, "gameday": "2026-11-29", "gametime": "13:00", "away": "ATL", "home": "MIN", "roof": "dome"},
    {"week": 12, "gameday": "2026-11-29", "gametime": "13:00", "away": "BAL", "home": "HOU", "roof": "dome"},
    {"week": 12, "gameday": "2026-11-29", "gametime": "16:05", "away": "TEN", "home": "JAX", "roof": "outdoors"},
    {"week": 12, "gameday": "2026-11-29", "gametime": "16:25", "away": "WAS", "home": "ARI", "roof": "dome"},
    {"week": 12, "gameday": "2026-11-29", "gametime": "16:25", "away": "SEA", "home": "SF", "roof": "outdoors"},
    {"week": 12, "gameday": "2026-11-29", "gametime": "20:20", "away": "NE", "home": "LAC", "roof": "outdoors"},
    {"week": 12, "gameday": "2026-11-30", "gametime": "20:15", "away": "CAR", "home": "TB", "roof": "outdoors"},
    {"week": 13, "gameday": "2026-12-03", "gametime": "20:15", "away": "KC", "home": "LA", "roof": "outdoors"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "13:00", "away": "DET", "home": "ATL", "roof": "dome"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "13:00", "away": "JAX", "home": "CHI", "roof": "outdoors"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "13:00", "away": "CIN", "home": "CLE", "roof": "outdoors"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "13:00", "away": "WAS", "home": "TEN", "roof": "outdoors"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "13:00", "away": "GB", "home": "NO", "roof": "dome"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "13:00", "away": "SF", "home": "NYG", "roof": "outdoors"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "13:00", "away": "LAC", "home": "TB", "roof": "outdoors"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "16:05", "away": "MIA", "home": "DEN", "roof": "outdoors"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "16:05", "away": "PHI", "home": "ARI", "roof": "dome"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "16:25", "away": "CAR", "home": "MIN", "roof": "dome"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "16:25", "away": "BUF", "home": "NE", "roof": "outdoors"},
    {"week": 13, "gameday": "2026-12-06", "gametime": "20:20", "away": "HOU", "home": "PIT", "roof": "outdoors"},
    {"week": 13, "gameday": "2026-12-07", "gametime": "20:15", "away": "DAL", "home": "SEA", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-10", "gametime": "20:15", "away": "MIN", "home": "NE", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "13:00", "away": "ATL", "home": "CLE", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "13:00", "away": "TEN", "home": "DET", "roof": "dome"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "13:00", "away": "CHI", "home": "MIA", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "13:00", "away": "DEN", "home": "NYJ", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "13:00", "away": "IND", "home": "PHI", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "13:00", "away": "HOU", "home": "WAS", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "13:00", "away": "NO", "home": "CAR", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "13:00", "away": "TB", "home": "BAL", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "16:05", "away": "LAC", "home": "LV", "roof": "dome"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "16:25", "away": "KC", "home": "CIN", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "16:25", "away": "LA", "home": "SF", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "16:25", "away": "NYG", "home": "SEA", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-13", "gametime": "20:20", "away": "BUF", "home": "GB", "roof": "outdoors"},
    {"week": 14, "gameday": "2026-12-14", "gametime": "20:15", "away": "PIT", "home": "JAX", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-17", "gametime": "20:15", "away": "SF", "home": "LAC", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-19", "gametime": "17:00", "away": "SEA", "home": "PHI", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-19", "gametime": "20:20", "away": "CHI", "home": "BUF", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "13:00", "away": "MIA", "home": "GB", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "13:00", "away": "IND", "home": "TEN", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "13:00", "away": "CLE", "home": "NYG", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "13:00", "away": "BAL", "home": "PIT", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "13:00", "away": "NO", "home": "TB", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "13:00", "away": "ATL", "home": "WAS", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "13:00", "away": "CIN", "home": "CAR", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "13:00", "away": "JAX", "home": "HOU", "roof": "dome"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "16:05", "away": "NYJ", "home": "ARI", "roof": "dome"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "16:25", "away": "DEN", "home": "LV", "roof": "dome"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "16:25", "away": "DAL", "home": "LA", "roof": "outdoors"},
    {"week": 15, "gameday": "2026-12-20", "gametime": "20:20", "away": "DET", "home": "MIN", "roof": "dome"},
    {"week": 15, "gameday": "2026-12-21", "gametime": "20:15", "away": "NE", "home": "KC", "roof": "outdoors"},
    {"week": 16, "gameday": "2026-12-24", "gametime": "20:15", "away": "HOU", "home": "PHI", "roof": "outdoors"},
    {"week": 16, "gameday": "2026-12-25", "gametime": "13:00", "away": "GB", "home": "CHI", "roof": "outdoors"},
    {"week": 16, "gameday": "2026-12-25", "gametime": "16:30", "away": "BUF", "home": "DEN", "roof": "outdoors"},
    {"week": 16, "gameday": "2026-12-25", "gametime": "20:15", "away": "LA", "home": "SEA", "roof": "outdoors"},
    {"week": 16, "gameday": "2026-12-27", "gametime": "00:00", "away": "TB", "home": "ATL", "roof": "dome"},
    {"week": 16, "gameday": "2026-12-27", "gametime": "00:00", "away": "CIN", "home": "IND", "roof": "dome"},
    {"week": 16, "gameday": "2026-12-27", "gametime": "00:00", "away": "WAS", "home": "MIN", "roof": "dome"},
    {"week": 16, "gameday": "2026-12-27", "gametime": "00:00", "away": "CAR", "home": "PIT", "roof": "outdoors"},
    {"week": 16, "gameday": "2026-12-27", "gametime": "13:00", "away": "LAC", "home": "MIA", "roof": "outdoors"},
    {"week": 16, "gameday": "2026-12-27", "gametime": "13:00", "away": "ARI", "home": "NO", "roof": "dome"},
    {"week": 16, "gameday": "2026-12-27", "gametime": "13:00", "away": "NE", "home": "NYJ", "roof": "outdoors"},
    {"week": 16, "gameday": "2026-12-27", "gametime": "13:00", "away": "CLE", "home": "BAL", "roof": "outdoors"},
    {"week": 16, "gameday": "2026-12-27", "gametime": "16:05", "away": "TEN", "home": "LV", "roof": "dome"},
    {"week": 16, "gameday": "2026-12-27", "gametime": "16:25", "away": "SF", "home": "KC", "roof": "outdoors"},
    {"week": 16, "gameday": "2026-12-27", "gametime": "20:20", "away": "JAX", "home": "DAL", "roof": "dome"},
    {"week": 16, "gameday": "2026-12-28", "gametime": "20:15", "away": "NYG", "home": "DET", "roof": "dome"},
    {"week": 17, "gameday": "2026-12-31", "gametime": "20:15", "away": "BAL", "home": "CIN", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "00:00", "away": "DEN", "home": "NE", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "00:00", "away": "KC", "home": "LAC", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "00:00", "away": "LA", "home": "TB", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "00:00", "away": "WAS", "home": "JAX", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "13:00", "away": "NO", "home": "ATL", "roof": "dome"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "13:00", "away": "IND", "home": "CLE", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "13:00", "away": "NYG", "home": "DAL", "roof": "dome"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "13:00", "away": "PIT", "home": "TEN", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "13:00", "away": "BUF", "home": "MIA", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "13:00", "away": "MIN", "home": "NYJ", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "13:00", "away": "SEA", "home": "CAR", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "16:05", "away": "LV", "home": "ARI", "roof": "dome"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "16:25", "away": "DET", "home": "CHI", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-03", "gametime": "20:20", "away": "PHI", "home": "SF", "roof": "outdoors"},
    {"week": 17, "gameday": "2027-01-04", "gametime": "20:15", "away": "HOU", "home": "GB", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "NYJ", "home": "BUF", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "CLE", "home": "CIN", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "LAC", "home": "DEN", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "DET", "home": "GB", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "JAX", "home": "IND", "roof": "dome"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "LV", "home": "KC", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "SEA", "home": "LA", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "CHI", "home": "MIN", "roof": "dome"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "MIA", "home": "NE", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "TB", "home": "NO", "roof": "dome"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "PHI", "home": "NYG", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "SF", "home": "ARI", "roof": "dome"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "DAL", "home": "WAS", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "ATL", "home": "CAR", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "PIT", "home": "BAL", "roof": "outdoors"},
    {"week": 18, "gameday": "2027-01-10", "gametime": "00:00", "away": "TEN", "home": "HOU", "roof": "dome"},
]


def build_upcoming_games(schedules: pd.DataFrame, odds_data: Optional[List], days_ahead: int = 120) -> List[Dict]:
    """
    Build upcoming games using the embedded 2026 official schedule as the
    primary source of truth (guarantees full weekly slates, e.g. Week 3 = 16).
    Odds / nflverse / ESPN only overlay lines and weather — they never remove games.
    """
    games: List[Dict] = []

    # Odds index for line overlay only
    odds_by_matchup: Dict[Tuple[str, str], Dict] = {}
    odds_by_date_teams: Dict[Tuple[str, str, str], Dict] = {}
    if odds_data:
        for ev in odds_data:
            h = _normalize_team_abbr(to_abbr(ev.get("home_team", "")) or "")
            a = _normalize_team_abbr(to_abbr(ev.get("away_team", "")) or "")
            if not h or not a:
                continue
            commence = (ev.get("commence_time") or "")[:10]
            odds_by_matchup[(h, a)] = ev
            if commence:
                odds_by_date_teams[(commence, h, a)] = ev

    today = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(days=200)  # full season window

    # ---- Primary: embedded official slate ----
    for row in EMBEDDED_2026_SCHEDULE:
        try:
            gameday = row["gameday"]
            gd = pd.to_datetime(gameday, errors="coerce")
            if pd.isna(gd):
                continue
            # Keep games from 2 days ago through cutoff (full weeks intact)
            if gd < today - pd.Timedelta(days=2):
                continue  # keep all future weeks through end of season
            home = _normalize_team_abbr(row["home"])
            away = _normalize_team_abbr(row["away"])
            week = int(row["week"])
            gametime = row.get("gametime") or "13:00"
            roof = row.get("roof") or "outdoors"

            odds_ev = odds_by_date_teams.get((gameday, home, away)) or odds_by_matchup.get((home, away))
            commence_raw = (odds_ev.get("commence_time") if odds_ev else "") or ""
            kickoff = format_schedule_kickoff(gameday, gametime)

            avg_spread, avg_total = _extract_odds_lines(odds_ev, home)
            if avg_total is None:
                avg_total = 45.0

            games.append({
                "week": week,
                "gameday": gameday,
                "gametime": gametime,
                "kickoff": kickoff,
                "home": home,
                "away": away,
                "home_full": full_name(home),
                "away_full": full_name(away),
                "roof": roof,
                "avg_spread": avg_spread,
                "avg_total": avg_total,
                "odds_event": odds_ev,
                "commence_raw": commence_raw or f"{gameday}T{gametime}:00Z",
                "game_id": f"2026_{week:02d}_{away}_{home}",
            })
        except Exception:
            continue

    # If embedded produced nothing (e.g. far future), fall back to schedule DF / odds
    if not games and schedules is not None and not getattr(schedules, "empty", True):
        try:
            sched = schedules.copy()
            if "gameday" in sched.columns:
                sched["_gd"] = pd.to_datetime(sched["gameday"], errors="coerce")
                sched = sched[sched["_gd"].notna()]
                sched = sched[(sched["_gd"] >= today - pd.Timedelta(days=2)) & (sched["_gd"] <= cutoff)]
            for _, row in sched.iterrows():
                home = _normalize_team_abbr(row.get("home_team", ""))
                away = _normalize_team_abbr(row.get("away_team", ""))
                if not home or not away:
                    continue
                gameday = str(row.get("gameday", ""))[:10]
                gametime = row.get("gametime") or "13:00"
                week = row.get("week")
                try:
                    week = int(week) if pd.notna(week) else estimate_week_from_date(gameday)
                except Exception:
                    week = estimate_week_from_date(gameday)
                odds_ev = odds_by_date_teams.get((gameday, home, away)) or odds_by_matchup.get((home, away))
                avg_spread, avg_total = _extract_odds_lines(odds_ev, home)
                if avg_total is None:
                    avg_total = 45.0
                games.append({
                    "week": week,
                    "gameday": gameday,
                    "gametime": gametime,
                    "kickoff": format_schedule_kickoff(gameday, gametime),
                    "home": home,
                    "away": away,
                    "home_full": full_name(home),
                    "away_full": full_name(away),
                    "roof": str(row.get("roof") or "outdoors").lower(),
                    "avg_spread": avg_spread,
                    "avg_total": avg_total,
                    "odds_event": odds_ev,
                    "commence_raw": (odds_ev.get("commence_time") if odds_ev else "") or f"{gameday}T{gametime}:00Z",
                    "game_id": row.get("game_id"),
                })
        except Exception:
            pass

    if not games and odds_data:
        games = build_upcoming_from_odds(odds_data, schedules if schedules is not None else pd.DataFrame())

    # Dedupe week+matchup
    seen = set()
    unique = []
    for g in games:
        key = (g.get("week"), g.get("home"), g.get("away"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(g)
    unique.sort(key=lambda g: (g.get("gameday") or "", str(g.get("gametime") or "")))
    return unique


# -----------------------------
# WEATHER
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_api(lat: float, lon: float, kickoff_iso: str) -> Dict[str, Any]:
    try:
        if not kickoff_iso or len(kickoff_iso) < 10:
            kickoff_iso = datetime.utcnow().strftime("%Y-%m-%dT17:00")
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "auto",
                "forecast_days": 14,
            },
            timeout=10,
        )
        if r.status_code != 200:
            return {"temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0, "source": f"http_{r.status_code}"}
        data = r.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        winds = hourly.get("wind_speed_10m", [])
        precs = hourly.get("precipitation_probability", [])
        if not times or not temps:
            return {"temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0, "source": "empty"}
        try:
            kick = pd.to_datetime(kickoff_iso)
            if kick.tzinfo is None:
                kick = kick.tz_localize("UTC")
        except Exception:
            kick = pd.Timestamp.utcnow()
        best_idx = 0
        best_diff = float("inf")
        for i, t in enumerate(times):
            try:
                tt = pd.to_datetime(t)
                if tt.tzinfo is None:
                    tt = tt.tz_localize("UTC")
                diff = abs((tt - kick).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            except Exception:
                continue
        return {
            "temp_f": float(temps[best_idx]),
            "wind_mph": float(winds[best_idx]) if best_idx < len(winds) else 5.0,
            "precip_prob": float(precs[best_idx]) if best_idx < len(precs) else 10.0,
            "source": "open-meteo",
        }
    except Exception as e:
        return {"temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0, "source": f"error:{type(e).__name__}"}

def make_weather_key(home: str, commence_raw: str) -> str:
    if commence_raw and len(commence_raw) >= 16:
        return f"{home}_{commence_raw[:16]}"
    if commence_raw and len(commence_raw) >= 10:
        return f"{home}_{commence_raw[:10]}"
    return f"{home}_{datetime.now().strftime('%Y-%m-%d')}"

def build_weather_cache_from_games(games: List[Dict]) -> Dict[str, Dict]:
    cache = {}
    real_count = 0
    fallback_count = 0
    samples = []
    if not games:
        return cache
    progress = st.progress(0, text="Fetching unique weather for each outdoor stadium...")
    total = len(games)
    for idx, g in enumerate(games):
        try:
            home = g["home"]
            if not home or home not in STADIUM_COORDS:
                continue
            commence_raw = g.get("commence_raw") or ""
            game_date = g.get("gameday") or (commence_raw[:10] if len(commence_raw) >= 10 else datetime.now().strftime("%Y-%m-%d"))
            roof = g.get("roof") or "outdoors"
            key = make_weather_key(home, commence_raw or game_date)
            if roof in ("dome", "closed"):
                cache[key] = {
                    "temp_f": 72.0, "wind_mph": 0.0, "precip_prob": 0.0,
                    "source": "dome", "roof": roof, "home": home
                }
            else:
                lat, lon = STADIUM_COORDS[home]
                wx = fetch_weather_api(lat, lon, commence_raw or f"{game_date}T17:00:00Z")
                wx["roof"] = roof
                wx["home"] = home
                cache[key] = wx
                if wx.get("source") == "open-meteo":
                    real_count += 1
                    if len(samples) < 8:
                        samples.append({
                            "team": home,
                            "temp": round(wx["temp_f"]),
                            "wind": round(wx["wind_mph"]),
                            "precip": round(wx["precip_prob"]),
                        })
                else:
                    fallback_count += 1
        except Exception:
            continue
        progress.progress((idx + 1) / total, text=f"Weather {idx+1}/{total}")
    progress.empty()
    st.session_state["weather_debug"] = {
        "real": real_count,
        "fallback": fallback_count,
        "samples": samples,
        "total_keys": len(cache)
    }
    return cache

def weather_adjustments(roof: str, weather: Dict) -> Dict[str, Any]:
    if roof in ("dome", "closed"):
        return {
            "total_adj": 0.0, "noise_extra": 0.0, "under_bias": 0.0,
            "rule_pts": 0.0, "label": "Dome / Closed"
        }
    temp = float(weather.get("temp_f", 70))
    wind = float(weather.get("wind_mph", 5))
    precip = float(weather.get("precip_prob", 10))
    total_adj = noise_extra = under_bias = rule_pts = 0.0
    labels = []
    if wind >= 20:
        total_adj -= 3.5; noise_extra += 2.5; under_bias += 0.04; rule_pts += 1.4
        labels.append(f"High wind {wind:.0f} mph")
    elif wind >= 15:
        total_adj -= 2.0; noise_extra += 1.5; under_bias += 0.025; rule_pts += 0.9
        labels.append(f"Wind {wind:.0f} mph")
    if precip >= 60:
        total_adj -= 2.5; noise_extra += 2.0; under_bias += 0.03; rule_pts += 1.1
        labels.append(f"Precip {precip:.0f}%")
    elif precip >= 40:
        total_adj -= 1.2; noise_extra += 1.0; under_bias += 0.015; rule_pts += 0.6
        labels.append(f"Precip {precip:.0f}%")
    if temp <= 25:
        total_adj -= 2.0; noise_extra += 1.5; rule_pts += 0.7
        labels.append(f"Very cold {temp:.0f}°F")
    elif temp <= 35:
        total_adj -= 1.0; noise_extra += 0.8; rule_pts += 0.4
        labels.append(f"Cold {temp:.0f}°F")
    elif temp >= 95:
        total_adj -= 1.0; noise_extra += 1.0; rule_pts += 0.4
        labels.append(f"Hot {temp:.0f}°F")
    label = " • ".join(labels) if labels else f"Outdoor {temp:.0f}°F / {wind:.0f} mph"
    return {
        "total_adj": total_adj, "noise_extra": noise_extra,
        "under_bias": under_bias, "rule_pts": rule_pts, "label": label
    }
# -----------------------------
# ML + MONTE CARLO
# -----------------------------
@st.cache_data(ttl=12 * 3600, show_spinner=False)
def prepare_historical_features(seasons: List[int]):
    try:
        sched = load_schedules(seasons)
        epa = get_team_epa(seasons)
        if sched.empty or epa.empty:
            return None
        completed = sched[
            sched["result"].notna() &
            sched["spread_line"].notna() &
            sched["home_score"].notna() &
            sched["away_score"].notna()
        ].copy()
        rows = []
        for _, row in completed.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            if home not in epa.index or away not in epa.index:
                continue
            home_off = float(epa.loc[home, "off_epa"])
            home_def = float(epa.loc[home, "def_epa"])
            away_off = float(epa.loc[away, "off_epa"])
            away_def = float(epa.loc[away, "def_epa"])
            epa_edge = (home_off - away_def) - (away_off - home_def)
            spread = float(row["spread_line"])
            result = float(row["result"])
            total_line = row.get("total_line", 45.0)
            if pd.isna(total_line):
                total_line = 45.0
            rows.append({
                "epa_edge": epa_edge, "spread": spread, "rest_diff": 0.0,
                "home_off": home_off, "home_def": home_def,
                "away_off": away_off, "away_def": away_def,
                "abs_spread": abs(spread), "total_line": float(total_line),
                "home_covered": 1 if result > spread else 0
            })
        df = pd.DataFrame(rows)
        return df if len(df) >= 80 else None
    except Exception:
        return None

@st.cache_resource(ttl=12 * 3600, show_spinner=False)
def train_ats_model(seasons: List[int]):
    hist = prepare_historical_features(seasons)
    if hist is None or hist.empty:
        return None
    feature_cols = [
        "epa_edge", "spread", "rest_diff",
        "home_off", "home_def", "away_off", "away_def",
        "abs_spread", "total_line"
    ]
    X = hist[feature_cols]
    y = hist["home_covered"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipe.fit(X, y)
    return pipe, feature_cols

def monte_carlo_game(
    home_off, home_def, away_off, away_def,
    spread, total_line, n_sims=8000,
    total_adj=0.0, noise_extra=0.0, under_bias=0.0,
    pace_adj=0.0, form_margin_adj=0.0
):
    expected_margin = (home_off - away_def - (away_off - home_def)) * 35.0 + 1.2 + form_margin_adj
    sim_margins = np.random.normal(expected_margin, 11.5 + noise_extra, n_sims)
    expected_total = 44.0 + (home_off + away_off - home_def - away_def) * 22.0 + total_adj + pace_adj
    sim_totals = np.random.normal(expected_total, 13.5 + noise_extra * 0.8, n_sims)
    home_cover = float(np.mean(sim_margins > spread))
    over_p = float(np.mean(sim_totals > total_line)) if total_line else 0.5
    over_p = max(0.05, min(0.95, over_p - under_bias))
    home_ev = home_cover * 100 / 110 - (1 - home_cover)
    away_ev = (1 - home_cover) * 100 / 110 - home_cover
    return {
        "home_cover_prob": home_cover,
        "over_prob": over_p,
        "under_prob": 1.0 - over_p,
        "home_ev": float(home_ev),
        "away_ev": float(away_ev)
    }
# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🎯 Game Signals",
    "📅 Games & Odds",
    "🎯 Player Props",
    "🌤️ Weather",
    "🏥 Injury Report",
    "📋 Depth Charts",
    "📊 Team History",
    "📘 Methodology",
    "⚙️ Advanced",
])
# ========== TAB 1 ==========
with tab1:
    st.subheader("Game Signals")
    with st.spinner("Loading EPA, Pace, Form, Schedule, Odds and unique weather..."):
        team_epa = get_team_epa()
        team_pace = get_team_pace()
        recent_form = get_recent_form(n_games=form_window)
        schedules = load_schedules()
        odds_data, odds_status = fetch_nfl_odds(api_key) if api_key else (None, "No API key entered")
        try:
            current_season = int(nfl.get_current_season())
        except Exception:
            current_season = datetime.now().year if datetime.now().month >= 8 else datetime.now().year - 1
        model_bundle = train_ats_model(list(range(current_season - 4, current_season)))
        # Source of truth: schedule-driven game list (includes every week 1–18 game)
        # Falls back to Odds API events if schedule rows are empty
        upcoming = build_upcoming_games(schedules, odds_data, days_ahead=90)
        stamp_now("schedule")
        if odds_data:
            stamp_now("odds")
        weather_cache = build_weather_cache_from_games(upcoming)
        stamp_now("weather")

    # Helpful diagnostics when the game list is empty
    if not upcoming:
        with st.expander("Schedule / odds diagnostics (why no games?)", expanded=True):
            st.write({
                "schedules_rows": 0 if schedules is None or schedules.empty else len(schedules),
                "schedules_columns": list(schedules.columns)[:12] if schedules is not None and not schedules.empty else [],
                "odds_events": 0 if not odds_data else len(odds_data),
                "api_key_present": bool(api_key),
                "current_season_detected": current_season,
            })
            if schedules is not None and not schedules.empty and "season" in schedules.columns:
                st.write("Seasons in schedule:", sorted(schedules["season"].dropna().unique().tolist()))
            if schedules is not None and not schedules.empty and "gameday" in schedules.columns:
                st.write("Gameday range:", str(schedules["gameday"].min()), "→", str(schedules["gameday"].max()))
            st.caption(
                "If schedules_rows is 0, update nflreadpy / clear cache. "
                "If odds_events is 0, enter a valid Odds API key. "
                "The app will use whichever source has data."
            )
    opportunities = []
    skipped = []
    if upcoming:
        model = model_bundle[0] if model_bundle else None
        feature_cols = model_bundle[1] if model_bundle else None
        league_avg_pace = float(team_pace["plays_per_game"].mean()) if not team_pace.empty else 65.0
        for g in upcoming:
            try:
                home = g["home"]
                away = g["away"]
                home_full = g["home_full"]
                away_full = g["away_full"]
                commence_raw = g.get("commence_raw") or ""
                commence = g.get("kickoff") or ""
                game_date = g.get("gameday") or (commence[:10] if commence else datetime.now().strftime("%Y-%m-%d"))
                roof = g.get("roof") or "outdoors"
                wx_key = make_weather_key(home, commence_raw or game_date)
                weather = weather_cache.get(wx_key) or {
                    "temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0,
                    "roof": roof, "source": "missing"
                }
                wx_adj = weather_adjustments(roof, weather)
                avg_spread = g.get("avg_spread")
                avg_total = g.get("avg_total") if g.get("avg_total") is not None else 45.0

                # Use team EPA when available; otherwise neutral league averages (never drop the game)
                if not team_epa.empty and home in team_epa.index and away in team_epa.index:
                    home_off = float(team_epa.loc[home, "off_epa"])
                    home_def = float(team_epa.loc[home, "def_epa"])
                    away_off = float(team_epa.loc[away, "off_epa"])
                    away_def = float(team_epa.loc[away, "def_epa"])
                else:
                    skipped.append(f"Neutral EPA used for {away} @ {home}")
                    home_off = home_def = away_off = away_def = 0.0
                epa_edge = (home_off - away_def) - (away_off - home_def)
                rest_diff = rest_differential(schedules, home, away, game_date, week=g.get("week"))
                # ---- SIGNALS ----
                if avg_spread is not None:
                    home_imp, away_imp = implied_team_totals(avg_spread, avg_total)
                else:
                    home_imp, away_imp = avg_total / 2, avg_total / 2
                home_form = recent_form.get(home, {"form_margin": 0.0, "form_epa": 0.0, "n": 0})
                away_form = recent_form.get(away, {"form_margin": 0.0, "form_epa": 0.0, "n": 0})
                form_margin_diff = home_form["form_margin"] - away_form["form_margin"]
                form_epa_diff = home_form["form_epa"] - away_form["form_epa"]
                home_pace = float(team_pace.loc[home, "plays_per_game"]) if (not team_pace.empty and home in team_pace.index) else league_avg_pace
                away_pace = float(team_pace.loc[away, "plays_per_game"]) if (not team_pace.empty and away in team_pace.index) else league_avg_pace
                combined_pace = (home_pace + away_pace) / 2.0
                pace_vs_avg = combined_pace - league_avg_pace
                pace_adj = pace_vs_avg * 0.35
                tz_diff = timezone_diff(home, away)
                travel_dir = travel_direction(home, away)
                div_flag = is_divisional(home, away)
                signals = []
                rule_score = 0.0
                if epa_edge > 0.08:
                    signals.append(f"Home EPA +{epa_edge:.3f}"); rule_score += 2.2
                elif epa_edge < -0.08:
                    signals.append(f"Away EPA {epa_edge:.3f}"); rule_score += 2.0
                if avg_spread is not None and avg_spread > 1.5:
                    signals.append("Home underdog"); rule_score += 1.3
                if avg_spread is not None and abs(avg_spread) >= 7:
                    signals.append(f"Large spread {avg_spread:+.1f}"); rule_score += 0.7
                if avg_total >= 48.5:
                    signals.append(f"High total {avg_total:.1f}"); rule_score += 0.6
                if rest_diff >= 3:
                    signals.append(f"Home rest +{rest_diff}d"); rule_score += 1.1
                elif rest_diff <= -3:
                    signals.append(f"Away rest {rest_diff}d"); rule_score += 1.0
                if wx_adj["rule_pts"] > 0:
                    signals.append(wx_adj["label"]); rule_score += wx_adj["rule_pts"]
                if home_imp >= 27.5:
                    signals.append(f"High Home Imp {home_imp:.1f}"); rule_score += 1.5
                elif home_imp <= 17.5:
                    signals.append(f"Low Home Imp {home_imp:.1f}"); rule_score += 1.2
                if away_imp >= 27.5:
                    signals.append(f"High Away Imp {away_imp:.1f}"); rule_score += 1.4
                elif away_imp <= 17.5:
                    signals.append(f"Low Away Imp {away_imp:.1f}"); rule_score += 1.1
                if avg_total >= 48 and (home_imp + away_imp) < 46:
                    signals.append("Implied soft total"); rule_score += 0.8
                if form_margin_diff >= 7:
                    signals.append(f"Home form +{form_margin_diff:.1f}"); rule_score += 1.6
                elif form_margin_diff <= -7:
                    signals.append(f"Away form {form_margin_diff:.1f}"); rule_score += 1.5
                if form_epa_diff > 0.12:
                    signals.append(f"Home form EPA +{form_epa_diff:.3f}"); rule_score += 1.3
                elif form_epa_diff < -0.12:
                    signals.append(f"Away form EPA {form_epa_diff:.3f}"); rule_score += 1.2
                if pace_vs_avg >= 4.0:
                    signals.append(f"Fast pace +{pace_vs_avg:.1f}"); rule_score += 1.0
                elif pace_vs_avg <= -4.0:
                    signals.append(f"Slow pace {pace_vs_avg:.1f}"); rule_score += 0.9
                if tz_diff >= 3:
                    if travel_dir == "Westbound":
                        signals.append(f"Away TZ -{tz_diff}h West"); rule_score += 1.1
                    else:
                        signals.append(f"Away TZ -{tz_diff}h East"); rule_score += 0.9
                elif tz_diff == 2:
                    signals.append(f"Away TZ -{tz_diff}h"); rule_score += 0.5
                if div_flag:
                    signals.append("Divisional"); rule_score += 0.7
                form_margin_adj = form_margin_diff * 0.15
                ml_home = 0.5
                if model is not None and avg_spread is not None and feature_cols is not None:
                    feat = pd.DataFrame([{
                        "epa_edge": epa_edge, "spread": avg_spread, "rest_diff": rest_diff,
                        "home_off": home_off, "home_def": home_def,
                        "away_off": away_off, "away_def": away_def,
                        "abs_spread": abs(avg_spread), "total_line": avg_total
                    }])[feature_cols]
                    ml_home = float(model.predict_proba(feat)[0, 1])
                mc = monte_carlo_game(
                    home_off, home_def, away_off, away_def,
                    avg_spread if avg_spread is not None else 0.0,
                    avg_total, n_sims=n_simulations,
                    total_adj=wx_adj["total_adj"],
                    noise_extra=wx_adj["noise_extra"],
                    under_bias=wx_adj["under_bias"],
                    pace_adj=pace_adj,
                    form_margin_adj=form_margin_adj
                )
                ml_edge = abs(ml_home - 0.5) * 4.0
                mc_edge = max(mc["home_ev"], mc["away_ev"]) * 8.0
                agree = 1.5 if ((ml_home > 0.5 and mc["home_cover_prob"] > 0.52) or
                                (ml_home < 0.5 and mc["home_cover_prob"] < 0.48)) else 0.0
                total_score = rule_score + ml_edge + mc_edge + agree
                if mc["home_ev"] > 0.03 and ml_home > 0.53:
                    rec = "Lean Home ATS"
                elif mc["away_ev"] > 0.03 and ml_home < 0.47:
                    rec = "Lean Away ATS"
                elif mc["over_prob"] > 0.56:
                    rec = "Lean Over"
                elif mc["under_prob"] > 0.56:
                    rec = "Lean Under"
                else:
                    rec = "No strong lean"

                # Market edge: blend ML + MC home cover/win vs market implied
                odds_ev = g.get("odds_event")
                model_home = 0.5 * ml_home + 0.5 * mc["home_cover_prob"]
                mkt_home = market_home_win_prob(odds_ev, home_full, away_full, avg_spread)
                edge_home = compute_edge(model_home, mkt_home)
                # Side edge aligned to recommendation
                if rec == "Lean Away ATS":
                    model_side = 1.0 - model_home
                    mkt_side = (1.0 - mkt_home) if mkt_home is not None else None
                    edge_pct = compute_edge(model_side, mkt_side)
                elif rec in ("Lean Over", "Lean Under"):
                    # totals edge vs 50/50 market baseline adjusted by under bias already in MC
                    if rec == "Lean Over":
                        edge_pct = (mc["over_prob"] - 0.5) * 100.0
                    else:
                        edge_pct = (mc["under_prob"] - 0.5) * 100.0
                else:
                    edge_pct = edge_home

                if roof in ("dome", "closed"):
                    wx_str = "Dome"
                else:
                    wx_str = (f"{weather.get('temp_f', 70):.0f}°F / "
                              f"{weather.get('wind_mph', 5):.0f} mph / "
                              f"{weather.get('precip_prob', 10):.0f}%")
                week_num = g.get("week")
                if week_num is None:
                    week_num = get_week(schedules, home, away, game_date)
                # Always include every scheduled game so weekly filters show the full slate
                opportunities.append({
                    "Week": week_num if week_num is not None else "—",
                    "Game": f"{away_full} @ {home_full}",
                    "Kickoff": commence,
                    "Roof": roof.title(),
                    "Weather": wx_str,
                    "Spread": f"{avg_spread:+.1f}" if avg_spread is not None else "—",
                    "Total": f"{avg_total:.1f}",
                    "Home Imp": f"{home_imp:.1f}",
                    "Away Imp": f"{away_imp:.1f}",
                    "EPA Edge": f"{epa_edge:+.3f}",
                    "Form Δ": f"{form_margin_diff:+.1f}",
                    "Pace": f"{combined_pace:.1f}",
                    "TZ Diff": f"{tz_diff}h" if tz_diff else "0",
                    "Div": "Yes" if div_flag else "No",
                    "Model %": f"{model_home*100:.1f}%",
                    "Market %": f"{mkt_home*100:.1f}%" if mkt_home is not None else "—",
                    "Edge %": f"{edge_pct:+.1f}" if edge_pct is not None else "—",
                    "ML Home %": f"{ml_home*100:.1f}%",
                    "MC Home %": f"{mc['home_cover_prob']*100:.1f}%",
                    "MC Over %": f"{mc['over_prob']*100:.1f}%",
                    "Recommendation": rec,
                    "Confidence": confidence_grade(
                        rec, total_score, ml_home, mc, edge_pct, len(signals), agree
                    ),
                    "Signals": " • ".join(signals) if signals else "—",
                    "Score": round(total_score, 2),
                    # hidden numeric helpers for tracker
                    "_model_prob": model_home,
                    "_market_prob": mkt_home if mkt_home is not None else None,
                    "_edge_pct": edge_pct,
                    "_spread": avg_spread,
                    "_total": avg_total,
                    "_home": home,
                    "_away": away,
                })
            except Exception as e:
                skipped.append(f"Error: {e}")
                continue
        if opportunities:
            try:
                _upsert_signals_from_opportunities(opportunities)
            except Exception:
                pass
            df = pd.DataFrame(opportunities).sort_values("Score", ascending=False)

            # ---- FILTER CONTROLS ----
            st.markdown("##### Filters")
            f1, f2, f3 = st.columns([1, 1.4, 1])
            with f1:
                min_score = st.slider(
                    "Min Score",
                    min_value=0.0,
                    max_value=max(10.0, float(df["Score"].max()) if len(df) else 10.0),
                    value=0.0,
                    step=0.5,
                    key="opp_min_score",
                )
            with f2:
                rec_options = {
                    "Home ATS": "Lean Home ATS",
                    "Away ATS": "Lean Away ATS",
                    "Over": "Lean Over",
                    "Under": "Lean Under",
                    "No strong lean": "No strong lean",
                }
                selected_recs = st.multiselect(
                    "Recommendation",
                    options=list(rec_options.keys()),
                    default=["Home ATS", "Away ATS", "Over", "Under", "No strong lean"],
                    key="opp_rec_filter",
                )
            with f3:
                df["_Week_num"] = pd.to_numeric(df["Week"], errors="coerce")
                available_weeks = sorted(df["_Week_num"].dropna().unique().tolist())
                week_choices = ["All weeks"] + [f"Week {int(w)}" for w in available_weeks]
                # Default to current NFL week when available
                cur_wk = current_nfl_week()
                default_week_idx = 0
                if cur_wk is not None and available_weeks:
                    label = f"Week {int(cur_wk)}"
                    if label in week_choices:
                        default_week_idx = week_choices.index(label)
                    else:
                        # nearest upcoming week in list
                        future = [w for w in available_weeks if w >= cur_wk]
                        pick = int(future[0]) if future else int(available_weeks[0])
                        label = f"Week {pick}"
                        if label in week_choices:
                            default_week_idx = week_choices.index(label)
                selected_week_filter = st.selectbox(
                    "Week",
                    options=week_choices,
                    index=default_week_idx,
                    key="opp_week_filter",
                )

            filtered = df[df["Score"] >= min_score].copy()
            if selected_recs:
                allowed = {rec_options[r] for r in selected_recs if r in rec_options}
                filtered = filtered[filtered["Recommendation"].isin(allowed)]
            if selected_week_filter != "All weeks" and available_weeks:
                try:
                    wk = int(selected_week_filter.replace("Week ", ""))
                    filtered = filtered[filtered["_Week_num"] == wk]
                except Exception:
                    pass

            # Week completeness vs official embedded slate
            week_note = ""
            if selected_week_filter != "All weeks":
                try:
                    wk = int(selected_week_filter.replace("Week ", ""))
                    expected_n = sum(1 for r in EMBEDDED_2026_SCHEDULE if int(r["week"]) == wk)
                    week_note = f" · Week {wk} official slate: **{expected_n}** games"
                    if len(filtered) < expected_n:
                        week_note += f" (showing {len(filtered)} after filters)"
                except Exception:
                    pass
            st.caption(
                f"Showing **{len(filtered)}** of **{len(df)}** games "
                f"(Min Score ≥ {min_score}"
                + (f", Week filter: {selected_week_filter}" if selected_week_filter != "All weeks" else "")
                + ")"
                + week_note
            )

            # ---- Top opportunity cards ----
            st.markdown("##### Top opportunities")
            card_n = min(5, len(filtered))
            if card_n:
                for i in range(card_n):
                    row = filtered.iloc[i]
                    conf = str(row.get("Confidence", "—"))
                    rec = str(row.get("Recommendation", "—"))
                    game = str(row.get("Game", "—"))
                    kick = str(row.get("Kickoff", "—"))
                    score = row.get("Score", "—")
                    edge = row.get("Edge %", "—")
                    spread = row.get("Spread", "—")
                    total = row.get("Total", "—")
                    signals = str(row.get("Signals", "—"))
                    with st.expander(f"{conf} · {rec} · {game}", expanded=(i == 0)):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Score", score)
                        c2.metric("Edge %", edge)
                        c3.metric("Spread", spread)
                        c4.metric("Total", total)
                        st.caption(f"Kickoff: {kick}")
                        st.write(signals)

            helper_cols = [c for c in filtered.columns if c.startswith("_")]
            display_df = filtered.drop(columns=["_Week_num"] + helper_cols, errors="ignore")
            # Color-ish confidence sort already by score
            st.markdown("##### Full board")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # ---- TOP 5 SIGNALED GAMES BY WEEK ----
            st.markdown("---")
            st.subheader("🏆 Top 5 Signaled Games by Week")
            st.caption(
                "Select a week from the dropdown to see its 5 highest-Score opportunities. "
                "Every scheduled game for that week is considered (schedule is source of truth)."
            )
            df_week = df.copy()
            df_week["Week_num"] = pd.to_numeric(df_week["Week"], errors="coerce")
            df_known = df_week[df_week["Week_num"].notna()].copy()
            display_cols = [
                "Game", "Kickoff", "Spread", "Total", "Home Imp", "Away Imp",
                "EPA Edge", "Form Δ", "Recommendation", "Confidence", "Score", "Signals"
            ]
            if not df_known.empty:
                weeks_sorted = sorted(df_known["Week_num"].unique())
                week_labels = {int(w): f"Week {int(w)}" for w in weeks_sorted}
                default_idx = 0
                cur_wk = current_nfl_week()
                if cur_wk is not None:
                    if int(cur_wk) in week_labels:
                        default_idx = list(weeks_sorted).index(
                            [w for w in weeks_sorted if int(w) == int(cur_wk)][0]
                        )
                    else:
                        future = [w for w in weeks_sorted if int(w) >= int(cur_wk)]
                        if future:
                            default_idx = list(weeks_sorted).index(future[0])
                selected_label = st.selectbox(
                    "Select week",
                    options=[week_labels[int(w)] for w in weeks_sorted],
                    index=default_idx,
                    key="top5_week_select",
                )
                selected_week = next(
                    int(w) for w, lab in week_labels.items() if lab == selected_label
                )
                week_df = (
                    df_known[df_known["Week_num"] == selected_week]
                    .sort_values("Score", ascending=False)
                    .head(5)
                )
                cols = [c for c in display_cols if c in week_df.columns]
                st.markdown(f"**{selected_label}** — top {len(week_df)} by Score")
                st.dataframe(week_df[cols], use_container_width=True, hide_index=True)
            else:
                st.warning(
                    "Could not resolve NFL week numbers from the schedule. "
                    "Showing overall Top 5 instead."
                )
                top5 = df.head(5)
                cols = [c for c in display_cols if c in top5.columns]
                st.dataframe(top5[cols], use_container_width=True, hide_index=True)

            st.markdown("#### Top Signal Summary")
            st.caption("Implied Team Totals, Recent Form, Pace, Travel/TZ and Divisional are folded into Score + Signals. Schedule is the source of truth for weeks and kickoff times.")
        else:
            st.warning("No opportunities matched the filters.")
            if skipped:
                with st.expander("Skipped"):
                    for s in skipped:
                        st.text(s)
    else:
        if not upcoming:
            st.warning(
                "No upcoming games found. Enter an Odds API key and/or ensure "
                "nflreadpy has the current season schedule (try Clear all caches)."
            )
        elif team_epa.empty:
            st.error("Could not load EPA data from nflreadpy.")
        else:
            st.warning("No opportunities to display.")

# ========== TAB 2 ==========
with tab2:
    st.subheader("Upcoming Games (full schedule)")
    st.caption(
        "Complete official slate from the embedded 2026 schedule. "
        "Every week lists every game with correct date/time. Odds fill in when available."
    )

    # Build display rows DIRECTLY from embedded schedule (never drop matchups)
    today = pd.Timestamp.now().normalize()
    rows = []
    for row in EMBEDDED_2026_SCHEDULE:
        try:
            gameday = row["gameday"]
            gd = pd.to_datetime(gameday, errors="coerce")
            # Show all remaining 2026 REG games (full season weeks 1-18)
            if pd.isna(gd) or gd < today - pd.Timedelta(days=2):
                continue
            home = row["home"]
            away = row["away"]
            week = int(row["week"])
            gametime = row.get("gametime") or "13:00"
            kickoff = format_schedule_kickoff(gameday, gametime)

            # Overlay odds from upcoming list if present
            avg_spread = avg_total = None
            match = next(
                (g for g in (upcoming or []) if g.get("home") == home and g.get("away") == away and g.get("week") == week),
                None,
            )
            if match:
                avg_spread = match.get("avg_spread")
                avg_total = match.get("avg_total")
            elif odds_data:
                # try odds API by team names
                for ev in odds_data:
                    h = to_abbr(ev.get("home_team", ""))
                    a = to_abbr(ev.get("away_team", ""))
                    if h == home and a == away:
                        s, t = _extract_odds_lines(ev, home)
                        avg_spread, avg_total = s, t
                        break

            spread = f"{avg_spread:+.1f}" if avg_spread is not None else "—"
            total = f"{avg_total:.1f}" if avg_total is not None else "—"
            imp_h = imp_a = "—"
            if avg_spread is not None and avg_total is not None:
                try:
                    ih, ia = implied_team_totals(avg_spread, avg_total)
                    imp_h, imp_a = f"{ih:.1f}", f"{ia:.1f}"
                except Exception:
                    pass
            rows.append({
                "Week": week,
                "Away": full_name(away),
                "Home": full_name(home),
                "Kickoff": kickoff,
                "Spread": spread,
                "Total": total,
                "Home Imp": imp_h,
                "Away Imp": imp_a,
                "Divisional": "Yes" if is_divisional(home, away) else "No",
                "Roof": str(row.get("roof") or "outdoors").title(),
            })
        except Exception:
            continue

    if rows:
        games_df = pd.DataFrame(rows)
        # Always offer every week that exists in the official embedded slate
        all_embed_weeks = sorted({int(r["week"]) for r in EMBEDDED_2026_SCHEDULE})
        weeks_in_view = sorted(games_df["Week"].dropna().unique().tolist())
        week_filter = st.selectbox(
            "Filter by week",
            options=["All weeks"] + [f"Week {int(w)}" for w in all_embed_weeks],
            key="tab2_week_filter",
        )
        display = games_df
        if week_filter != "All weeks":
            try:
                wk = int(week_filter.replace("Week ", ""))
                display = games_df[games_df["Week"] == wk].copy()
                # If anything missing for this week, rebuild purely from embed
                expected = [r for r in EMBEDDED_2026_SCHEDULE if int(r["week"]) == wk]
                if len(display) < len(expected):
                    rebuilt = []
                    for r in expected:
                        home, away = r["home"], r["away"]
                        kickoff = format_schedule_kickoff(r["gameday"], r.get("gametime") or "13:00")
                        existing = display[
                            (display["Home"] == full_name(home)) & (display["Away"] == full_name(away))
                        ] if not display.empty else display
                        if not existing.empty:
                            rebuilt.append(existing.iloc[0].to_dict())
                        else:
                            rebuilt.append({
                                "Week": wk,
                                "Away": full_name(away),
                                "Home": full_name(home),
                                "Kickoff": kickoff,
                                "Spread": "—",
                                "Total": "—",
                                "Home Imp": "—",
                                "Away Imp": "—",
                                "Divisional": "Yes" if is_divisional(home, away) else "No",
                                "Roof": str(r.get("roof") or "outdoors").title(),
                            })
                    display = pd.DataFrame(rebuilt)
            except Exception:
                pass
        st.dataframe(display, use_container_width=True, hide_index=True)
        counts = games_df.groupby("Week").size().sort_index()
        count_str = " · ".join([f"W{int(w)}:{int(n)}" for w, n in counts.items()])
        st.caption(f"{len(display)} games shown · Full slate counts: {count_str}")
        # Explicit Week 3 checklist
        w3 = games_df[games_df["Week"] == 3]
        if not w3.empty:
            has_nejax = ((w3["Away"].str.contains("New England")) & (w3["Home"].str.contains("Jacksonville"))).any()
            has_phichi = ((w3["Away"].str.contains("Philadelphia")) & (w3["Home"].str.contains("Chicago"))).any()
            st.info(
                f"Week 3 verification: **{len(w3)}/16 games** · "
                f"NE @ JAX: {'✅' if has_nejax else '❌'} · "
                f"PHI @ CHI: {'✅' if has_phichi else '❌'}"
            )
    else:
        st.warning("No upcoming games in the embedded schedule window.")


with tab3:
    st.subheader("Player Props")



    if not api_key:
        st.warning("Enter API key first.")
    elif not odds_data:
        st.info("No games with live odds available.")
    else:
        options = {f"{g.get('away_team')} @ {g.get('home_team')}": g.get("id") for g in odds_data}
        selected = st.selectbox("Select game", list(options.keys()))
        if st.button("Load Player Props", type="primary"):
            with st.spinner("Fetching..."):
                props = fetch_player_props(api_key, options[selected])
            if not props:
                st.error("Failed to fetch")
            elif "error" in props:
                st.error(props.get("error"))
                st.caption("Player props usually require a paid plan.")
            else:
                rows = []
                for book in props.get("bookmakers", []):
                    for market in book.get("markets", []):
                        for o in market.get("outcomes", []):
                            rows.append({
                                "Book": book.get("title"),
                                "Market": (market.get("key") or "").replace("player_", "").replace("_", " ").title(),
                                "Player": o.get("description") or o.get("name"),
                                "Side": o.get("name"),
                                "Line": o.get("point"),
                                "Odds": o.get("price")
                            })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.warning("No props returned.")

    # ========== TAB 4 ==========

with tab4:
    st.subheader("Game Weather")
    st.caption(
        "Forecast at each outdoor stadium near kickoff (Open-Meteo). "
        "Domes show controlled conditions. Used for total adjustments and under-bias in the model."
    )
    debug = st.session_state.get("weather_debug", {})
    if debug:
        st.caption(
            f"Real Open-Meteo pulls: **{debug.get('real', 0)}** · "
            f"Fallbacks: **{debug.get('fallback', 0)}** · Keys: **{debug.get('total_keys', 0)}**"
        )
    wx_rows = []
    if upcoming:
        for g in upcoming:
            home = g.get("home")
            away = g.get("away")
            roof = (g.get("roof") or "outdoors").lower()
            commence_raw = g.get("commence_raw") or ""
            game_date = g.get("gameday") or (commence_raw[:10] if len(str(commence_raw)) >= 10 else "")
            key = make_weather_key(home, commence_raw or game_date)
            weather = (weather_cache or {}).get(key) or {}
            if roof in ("dome", "closed"):
                temp = 72.0
                wind = 0.0
                precip = 0.0
                source = "dome"
                wx_label = "Dome / Closed"
            else:
                temp = float(weather.get("temp_f", 70) or 70)
                wind = float(weather.get("wind_mph", 5) or 5)
                precip = float(weather.get("precip_prob", 10) or 10)
                source = weather.get("source", "—")
                wx_adj = weather_adjustments(roof, weather)
                wx_label = wx_adj.get("label") or "Outdoor"
            wx_rows.append({
                "Week": g.get("week") if g.get("week") is not None else "—",
                "Game": f"{g.get('away_full') or away} @ {g.get('home_full') or home}",
                "Kickoff": g.get("kickoff") or game_date,
                "Stadium": home,
                "Roof": str(roof).title(),
                "Temp (°F)": round(temp),
                "Wind (mph)": round(wind),
                "Precip %": round(precip),
                "Impact": wx_label if roof not in ("dome", "closed") else "None (dome)",
                "Source": source,
            })
    if wx_rows:
        wx_df = pd.DataFrame(wx_rows)
        weeks = sorted({w for w in wx_df["Week"].tolist() if w != "—"})
        week_sel = st.selectbox(
            "Filter by week",
            options=["All weeks"] + [f"Week {int(w)}" for w in weeks],
            key="wx_week_filter",
        )
        show = wx_df
        if week_sel != "All weeks":
            try:
                wk = int(week_sel.replace("Week ", ""))
                show = wx_df[wx_df["Week"] == wk]
            except Exception:
                pass
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(f"{len(show)} games shown")
    else:
        st.info("No upcoming games / weather available yet. Load Game Signals first so weather is fetched.")


with tab5:
    st.subheader("NFL Injury Report")
    st.caption(
        "Official report from [NFL.com/injuries](https://www.nfl.com/injuries/). "
        "Filter by team. Game Status reflects the league designation (Out / Doubtful / Questionable / etc.)."
    )
    with st.spinner("Loading NFL.com injury report..."):
        inj_df = load_nfl_injury_report()
        if inj_df is not None and not inj_df.empty:
            stamp_now("injuries")
    if inj_df is None or inj_df.empty:
        st.warning(
            "Could not load injury data from NFL.com or ESPN right now. "
            "Click **Clear all caches** in the sidebar, then reload this tab. "
            "You can also open https://www.nfl.com/injuries/ directly."
        )
    else:
        teams = sorted(inj_df["Team"].dropna().unique().tolist())
        c1, c2, c3 = st.columns([1.4, 1, 1])
        with c1:
            team_sel = st.multiselect(
                "Team",
                options=teams,
                default=[],
                placeholder="All teams",
                key="inj_team_filter",
            )
        with c2:
            statuses = sorted({s for s in inj_df["Game Status"].dropna().unique().tolist() if s and s != "—"})
            status_sel = st.multiselect(
                "Game Status",
                options=statuses,
                default=[],
                placeholder="All statuses",
                key="inj_status_filter",
            )
        with c3:
            positions = sorted({p for p in inj_df["Position"].dropna().unique().tolist() if p})
            pos_sel = st.multiselect(
                "Position",
                options=positions,
                default=[],
                placeholder="All positions",
                key="inj_pos_filter",
            )
        view = inj_df.copy()
        if team_sel:
            view = view[view["Team"].isin(team_sel)]
        if status_sel:
            view = view[view["Game Status"].isin(status_sel)]
        if pos_sel:
            view = view[view["Position"].isin(pos_sel)]
        # Highlight OUT / Doubtful
        st.dataframe(
            view.drop(columns=["Team Abbr"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"{len(view)} players shown · Source: nfl.com/injuries")
        outs = view[view["Game Status"].astype(str).str.lower().isin(["out", "doubtful"])]
        if not outs.empty:
            st.markdown("##### Out / Doubtful")
            st.dataframe(
                outs.drop(columns=["Team Abbr"], errors="ignore"),
                use_container_width=True,
                hide_index=True,
            )


with tab6:
    st.subheader("Depth Charts")
    st.caption(
        "Current team depth charts from [Ourlads](https://www.ourlads.com/nfldepthcharts/). "
        "Pick a team to view starters first, then full depth."
    )
    with st.spinner("Loading depth charts..."):
        dc_df = load_depth_charts()
        if dc_df is not None and not dc_df.empty:
            stamp_now("depth")
    if dc_df is None or dc_df.empty:
        st.warning("Could not load depth charts right now. Try clearing caches and reloading.")
    else:
        teams = sorted(dc_df["Team"].dropna().unique().tolist())
        c1, c2 = st.columns([2, 1])
        with c1:
            team_sel = st.selectbox("Team", options=teams, key="dc_team")
        with c2:
            units = ["All"] + sorted(dc_df["Unit"].dropna().unique().tolist())
            unit_sel = st.selectbox("Unit", options=units, key="dc_unit")

        view = dc_df[dc_df["Team"] == team_sel].copy()
        if unit_sel != "All":
            view = view[view["Unit"] == unit_sel]

        if view.empty:
            st.info("No depth chart rows for this selection.")
        else:
            # ---- Starters at top ----
            st.markdown(f"##### Starters — {team_sel}")
            starters = (
                view[view["Rank"] == 1][["Unit", "Position", "Player"]]
                .sort_values(["Unit", "Position"])
                .reset_index(drop=True)
            )
            st.dataframe(starters, use_container_width=True, hide_index=True)
            st.caption(f"{len(starters)} starters")

            st.markdown("---")
            st.markdown(f"##### Full depth chart — {team_sel}")
            show = (
                view[["Unit", "Position", "Rank", "Player"]]
                .sort_values(["Unit", "Position", "Rank"])
                .reset_index(drop=True)
            )
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption(f"{len(show)} entries · source: Ourlads")



with tab7:
    st.subheader("Team History")
    st.caption(
        "ATS (against the spread) and Over/Under records by team — last 5 seasons of completed games. "
        "Filter by team and year."
    )
    with st.spinner("Loading team history (5 seasons)..."):
        th = build_team_history()
    if th is None or th.empty:
        st.warning("Could not load historical schedule results. Try Clear all caches.")
    else:
        teams = sorted(th["team"].dropna().unique().tolist())
        years = sorted([int(y) for y in th["season"].dropna().unique().tolist()], reverse=True)
        c1, c2 = st.columns(2)
        with c1:
            team_sel = st.selectbox(
                "Team",
                options=["All teams"] + [f"{full_name(t)} ({t})" for t in teams],
                key="th_team",
            )
        with c2:
            year_sel = st.selectbox(
                "Year",
                options=["All years"] + [str(y) for y in years],
                key="th_year",
            )
        view = th.copy()
        if team_sel != "All teams":
            abbr = team_sel.split("(")[-1].replace(")", "").strip()
            view = view[view["team"] == abbr]
        if year_sel != "All years":
            view = view[view["season"] == int(year_sel)]

        # Summary metrics
        ats = view[view["ats"].isin(["Cover", "Not Cover"])]
        covers = int((ats["ats"] == "Cover").sum())
        ncovers = int((ats["ats"] == "Not Cover").sum())
        ats_n = covers + ncovers
        ou = view[view["ou"].isin(["Over", "Under"])]
        # For team filter, each game appears once for that team so OU is fine;
        # for All teams each game appears twice — dedupe for OU summary
        if team_sel == "All teams":
            ou_dedupe = view.drop_duplicates(subset=["season", "gameday", "team", "opponent"])
            # still double - use home only
            ou_dedupe = view[view["home_away"] == "Home"]
            ou = ou_dedupe[ou_dedupe["ou"].isin(["Over", "Under"])]
        overs = int((ou["ou"] == "Over").sum())
        unders = int((ou["ou"] == "Under").sum())
        ou_n = overs + unders

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ATS", f"{covers}-{ncovers}", delta=f"{covers/ats_n:.0%} cover" if ats_n else None)
        m2.metric("ATS games", ats_n)
        m3.metric("O/U", f"{overs}-{unders}", delta=f"{overs/ou_n:.0%} over" if ou_n else None)
        m4.metric("O/U games", ou_n)

        # By season breakdown when all years
        if year_sel == "All years" and not view.empty:
            st.markdown("##### By season")
            season_rows = []
            for season, grp in view.groupby("season"):
                a = grp[grp["ats"].isin(["Cover", "Not Cover"])]
                c = int((a["ats"] == "Cover").sum())
                nc = int((a["ats"] == "Not Cover").sum())
                ogrp = grp if team_sel != "All teams" else grp[grp["home_away"] == "Home"]
                o = ogrp[ogrp["ou"].isin(["Over", "Under"])]
                ov = int((o["ou"] == "Over").sum())
                un = int((o["ou"] == "Under").sum())
                season_rows.append({
                    "Season": int(season),
                    "ATS": f"{c}-{nc}",
                    "ATS Cover %": f"{c/(c+nc):.0%}" if (c+nc) else "—",
                    "O/U": f"{ov}-{un}",
                    "Over %": f"{ov/(ov+un):.0%}" if (ov+un) else "—",
                })
            st.dataframe(pd.DataFrame(season_rows).sort_values("Season", ascending=False), use_container_width=True, hide_index=True)

        st.markdown("##### Game log")
        log = view.copy()
        log["Team"] = log["team"].map(lambda a: full_name(a) if a else a)
        log["Opponent"] = log["opponent"].map(lambda a: full_name(a) if a else a)
        show = log[[
            "season", "week", "gameday", "Team", "home_away", "Opponent",
            "spread", "margin", "ats", "total_line", "total_pts", "ou"
        ]].rename(columns={
            "season": "Season", "week": "Week", "gameday": "Date",
            "home_away": "H/A", "spread": "Spread", "margin": "Margin",
            "ats": "ATS", "total_line": "Total Line", "total_pts": "Points", "ou": "O/U",
        }).sort_values(["Season", "Date"], ascending=[False, False])
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(f"{len(show)} team-games · spreads/totals from historical schedule lines when available")


with tab8:
    st.subheader("Methodology")
    st.caption("How Score, Confidence, and Lean recommendations are produced. Research tool only — not betting advice.")

    st.markdown("### Lean (recommendation)")
    st.markdown(
        """
Leans are assigned in this order (first match wins):

1. **Lean Home ATS** — Monte Carlo home EV > 0.03 **and** logistic model P(home covers) > 0.53
2. **Lean Away ATS** — away EV > 0.03 **and** model P(home covers) < 0.47
3. **Lean Over** — simulated over probability > 0.56
4. **Lean Under** — simulated under probability > 0.56
5. **No strong lean** — none of the above

ATS leans require **both** positive simulated EV at −110 **and** model confidence past those thresholds. Totals only require the Monte Carlo probability gate.
        """
    )

    st.markdown("### Score")
    st.markdown(
        r"""
\[
\textbf{Score} = \text{rule\_score} + \text{ml\_edge} + \text{mc\_edge} + \text{agree}
\]

- **rule_score** — sum of heuristic signal points (EPA edge, rest, form, weather, implied totals, pace, travel, divisional, etc.). Form and rest use **current season only** (Week 1 rest advantage is forced to 0).
- **ml_edge** — \(|P_{\text{model}}(\text{home covers}) - 0.5| \times 4\)
- **mc_edge** — \(\max(\text{home EV},\ \text{away EV}) \times 8\) from Monte Carlo at −110 prices
- **agree** — +1.5 when the logistic model and Monte Carlo lean the same side

Higher Score means more independent support and stronger model/MC agreement — it is **not** a calibrated win probability.
        """
    )

    st.markdown("### Confidence (A / B / C / D / F)")
    st.markdown(
        """
Confidence grades the **strength of the lean** (no **E** grade):

| Grade | Meaning |
|-------|---------|
| **A** | Strong score + clear lean + model/MC agreement + solid edge |
| **B** | Strong overall with minor gaps |
| **C** | Decent lean, moderate evidence |
| **D** | Weak lean, or signals without a formal lean |
| **F** | No strong lean / minimal support |

Built from total Score, side probability vs 50%, model–MC agreement, market edge %, signal count, and simulated EV.
        """
    )

    st.markdown("### Monte Carlo (feeds lean + score)")
    st.markdown(
        r"""
**Margin**

\[
\mathbb{E}[\text{margin}] = (\text{home\_off}-\text{away\_def}-\text{away\_off}+\text{home\_def})\times 35 + 1.2 + \text{form adjustment}
\]

Simulated margins \(\sim \mathcal{N}(\mathbb{E}[\text{margin}],\ 11.5 + \text{weather noise})\).  
Home cover probability = share of draws beating the spread.

**Total**

\[
\mathbb{E}[\text{total}] = 44 + \text{EPA total factor} + \text{weather adj} + \text{pace adj}
\]

Over/under probabilities are taken from simulated totals vs the market line (with optional under-bias in poor weather).
        """
    )

    st.markdown("### Market Edge %")
    st.markdown(
        """
**Model %** blends logistic + Monte Carlo home probability.  
**Market %** is the fair (vig-removed) moneyline probability when available, otherwise a spread-based approximation.  
**Edge %** is model − market on the lean side (percentage points).
        """
    )

    st.markdown("### Data sources")
    st.markdown(
        """
- Schedule: embedded official slate + nflverse / ESPN  
- Odds: The Odds API  
- EPA / pace / form: nflreadpy play-by-play (form = current season only)  
- Weather: Open-Meteo by stadium + kickoff  
- Injuries: NFL.com (ESPN fallback)  
- Depth charts: Ourlads  
        """
    )

with tab9:
    st.subheader("Advanced")
    st.caption("Less frequently used tools — props, bankroll tracking, and historical backtests.")
    adv = st.radio(
        "Section",
        options=["Signal History", "Bankroll & CLV", "Backtest"],
        horizontal=True,
        key="advanced_section",
    )
    st.markdown("---")
    if adv == "Signal History":
        st.markdown("##### Signal History")
        st.caption(
            "Tracks Game Signals recommendations and grades them when results are in. "
            "Sorted by confidence. Record by grade (e.g. D: 0-1) counts Correct-Incorrect (pushes excluded)."
        )
        # Ensure we grade against schedule
        try:
            sched_for_grade = load_schedules()
        except Exception:
            sched_for_grade = pd.DataFrame()
        hist = _grade_signal_history(sched_for_grade)
        hist = _load_signal_history()

        if hist is None or hist.empty:
            st.info(
                "No signals logged yet. Open **Game Signals** so recommendations are saved, "
                "then return here after games complete to see graded results."
            )
        else:
            # Only truly graded rows (Correct/Incorrect). Pending/Push/N/A excluded.
            graded = hist[hist["result"].isin(["Correct", "Incorrect"])].copy()
            conf_order = ["A", "B", "C", "D", "F"]
            rec_order = ["Lean Home ATS", "Lean Away ATS", "Lean Over", "Lean Under"]
            summary_rows = []
            detail_rows = []
            if not graded.empty:
                graded["confidence"] = graded["confidence"].astype(str).str.upper().str[:1]
                for conf in conf_order:
                    sub = graded[graded["confidence"] == conf]
                    if sub.empty:
                        continue
                    wins = int((sub["result"] == "Correct").sum())
                    losses = int((sub["result"] == "Incorrect").sum())
                    total = wins + losses
                    summary_rows.append({
                        "Confidence": conf,
                        "Record": f"{wins}-{losses}",
                        "Win %": f"{wins / total:.0%}" if total else "—",
                        "N": total,
                    })
                    for rec in rec_order:
                        rsub = sub[sub["recommendation"].astype(str) == rec]
                        if rsub.empty:
                            continue
                        rw = int((rsub["result"] == "Correct").sum())
                        rl = int((rsub["result"] == "Incorrect").sum())
                        rt = rw + rl
                        detail_rows.append({
                            "Confidence": conf,
                            "Recommendation": rec.replace("Lean ", ""),
                            "Record": f"{rw}-{rl}",
                            "Win %": f"{rw / rt:.0%}" if rt else "—",
                            "N": rt,
                        })
                ow = int((graded["result"] == "Correct").sum())
                ol = int((graded["result"] == "Incorrect").sum())
                summary_rows.append({
                    "Confidence": "ALL",
                    "Record": f"{ow}-{ol}",
                    "Win %": f"{ow / (ow + ol):.0%}" if (ow + ol) else "—",
                    "N": ow + ol,
                })
            pending_n = int((hist["result"].astype(str) == "Pending").sum()) if not hist.empty else 0
            st.caption(f"Graded completed games only · **{pending_n}** still Pending")
            if summary_rows:
                st.markdown("**Record by confidence**")
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            if detail_rows:
                st.markdown("**By confidence × recommendation type**")
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
            if not summary_rows:
                st.info("No completed/graded signals yet. Only finished games appear in the records above.")

            # Full table sorted by confidence then date
            show = hist.copy()
            conf_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
            show["_cr"] = show["confidence"].astype(str).str.upper().str[:1].map(lambda x: conf_rank.get(x, 9))
            show = show.sort_values(["_cr", "week", "logged_at"], ascending=[True, True, False])
            display_cols = [
                c for c in [
                    "confidence", "recommendation", "result", "week", "game", "kickoff",
                    "score", "spread", "total", "logged_at", "graded_at",
                ] if c in show.columns
            ]
            st.markdown("**All signals**")
            st.dataframe(
                show[display_cols].rename(columns={
                    "confidence": "Confidence",
                    "recommendation": "Recommendation",
                    "result": "Result",
                    "week": "Week",
                    "game": "Game",
                    "kickoff": "Kickoff",
                    "score": "Score",
                    "spread": "Spread",
                    "total": "Total",
                    "logged_at": "Logged",
                    "graded_at": "Graded",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"{len(show)} signals · Pending rows grade automatically when final scores are available in the schedule.")
            st.download_button(
                "Download signal history CSV",
                data=hist.to_csv(index=False),
                file_name="signal_history.csv",
                mime="text/csv",
                key="signal_hist_dl",
            )
            if st.button("Clear signal history", key="signal_hist_clear"):
                _save_signal_history(pd.DataFrame(columns=hist.columns))
                st.rerun()

    elif adv == "Bankroll & CLV":
        st.markdown("##### Bankroll & Closing Line Value")
        st.caption(
            "Log units on model leans, grade results, and track CLV (closing line value). "
            "Positive CLV means you beat the closing number — the best long-term skill metric."
        )
        bet_df = _load_bet_log()

        # ---- Quick-add from a lean ----
        st.markdown("##### Log a bet")
        c1, c2, c3 = st.columns(3)
        with c1:
            b_game = st.text_input("Game", value="", placeholder="Away @ Home", key="bet_game")
            b_week = st.number_input("Week", min_value=1, max_value=22, value=1, key="bet_week")
            b_type = st.selectbox("Bet type", ["ATS", "Total", "ML"], key="bet_type")
        with c2:
            b_side = st.selectbox("Side", ["Home", "Away", "Over", "Under"], key="bet_side")
            b_line = st.number_input("Line taken", value=0.0, step=0.5, format="%.1f", key="bet_line")
            b_odds = st.number_input("Odds (American)", value=-110, step=5, key="bet_odds")
        with c3:
            b_units = st.number_input("Units", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="bet_units")
            b_model = st.number_input("Model prob (0-1)", min_value=0.0, max_value=1.0, value=0.55, step=0.01, key="bet_model")
            b_mkt = st.number_input("Market prob (0-1)", min_value=0.0, max_value=1.0, value=0.50, step=0.01, key="bet_mkt")
        b_notes = st.text_input("Notes", key="bet_notes")
        if st.button("Add to log", type="primary", key="bet_add"):
            import uuid
            edge = (b_model - b_mkt) * 100.0
            new_row = {
                "id": str(uuid.uuid4())[:8],
                "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "week": int(b_week),
                "game": b_game,
                "bet_type": b_type,
                "side": b_side,
                "line_taken": float(b_line),
                "odds": float(b_odds),
                "units": float(b_units),
                "model_prob": float(b_model),
                "market_prob": float(b_mkt),
                "edge_pct": round(edge, 2),
                "closing_line": None,
                "result": "Pending",
                "profit_units": 0.0,
                "clv": None,
                "notes": b_notes,
            }
            bet_df = pd.concat([bet_df, pd.DataFrame([new_row])], ignore_index=True)
            _save_bet_log(bet_df)
            st.success("Bet logged.")
            st.rerun()

        st.markdown("---")
        st.markdown("##### Open & settled bets")
        if bet_df is None or bet_df.empty:
            st.info("No bets logged yet. Add one above, or use Edge % from Game Signals to size spots.")
        else:
            st.dataframe(bet_df.drop(columns=["id"], errors="ignore"), use_container_width=True, hide_index=True)

            st.markdown("##### Grade / update a bet")
            ids = bet_df["id"].astype(str).tolist() if "id" in bet_df.columns else []
            if ids:
                pick = st.selectbox("Bet id", ids, key="bet_grade_id")
                row = bet_df[bet_df["id"].astype(str) == pick].iloc[0]
                g1, g2, g3, g4 = st.columns(4)
                with g1:
                    close_line = st.number_input(
                        "Closing line",
                        value=float(row["closing_line"]) if pd.notna(row.get("closing_line")) else float(row.get("line_taken") or 0),
                        step=0.5,
                        format="%.1f",
                        key="bet_close",
                    )
                with g2:
                    result = st.selectbox(
                        "Result",
                        ["Pending", "Win", "Loss", "Push"],
                        index=["Pending", "Win", "Loss", "Push"].index(str(row.get("result") or "Pending"))
                        if str(row.get("result") or "Pending") in ["Pending", "Win", "Loss", "Push"] else 0,
                        key="bet_result",
                    )
                with g3:
                    st.write(f"Line taken: **{row.get('line_taken')}**")
                    st.write(f"Side: **{row.get('side')}** · Type: **{row.get('bet_type')}**")
                with g4:
                    if st.button("Save grade", key="bet_save_grade"):
                        idx = bet_df.index[bet_df["id"].astype(str) == pick][0]
                        bet_df.at[idx, "closing_line"] = close_line
                        bet_df.at[idx, "result"] = result
                        side = str(row.get("side") or "Home").lower()
                        btype = str(row.get("bet_type") or "ATS").upper()
                        lt = float(row.get("line_taken") or 0)
                        if btype == "TOTAL":
                            clv = clv_total(lt, close_line, "over" if "over" in side else "under")
                        else:
                            clv = clv_spread(lt, close_line, "home" if "home" in side else "away")
                        bet_df.at[idx, "clv"] = round(clv, 2)
                        units = float(row.get("units") or 1)
                        odds = float(row.get("odds") or -110)
                        if result == "Win":
                            bet_df.at[idx, "profit_units"] = round(american_profit(units, odds, True), 3)
                        elif result == "Loss":
                            bet_df.at[idx, "profit_units"] = round(american_profit(units, odds, False), 3)
                        elif result == "Push":
                            bet_df.at[idx, "profit_units"] = 0.0
                        _save_bet_log(bet_df)
                        st.success(f"Updated. CLV = {clv:+.1f} pts")
                        st.rerun()

            # Summary metrics
            settled = bet_df[bet_df["result"].isin(["Win", "Loss", "Push"])] if "result" in bet_df.columns else bet_df.iloc[0:0]
            st.markdown("##### Performance")
            m1, m2, m3, m4 = st.columns(4)
            if len(settled):
                wins = (settled["result"] == "Win").sum()
                losses = (settled["result"] == "Loss").sum()
                decided = wins + losses
                wr = wins / decided if decided else 0
                profit = settled["profit_units"].sum() if "profit_units" in settled.columns else 0
                clv_avg = settled["clv"].mean() if "clv" in settled.columns and settled["clv"].notna().any() else None
                m1.metric("Record", f"{wins}-{losses}", delta=f"{wr:.1%} win" if decided else None)
                m2.metric("Profit (u)", f"{profit:+.2f}")
                m3.metric("Avg CLV (pts)", f"{clv_avg:+.2f}" if clv_avg is not None and pd.notna(clv_avg) else "—")
                m4.metric("Bets graded", f"{len(settled)}")
            else:
                m1.metric("Record", "0-0")
                m2.metric("Profit (u)", "0.00")
                m3.metric("Avg CLV (pts)", "—")
                m4.metric("Bets graded", "0")

            st.download_button(
                "Download bet log CSV",
                data=bet_df.to_csv(index=False),
                file_name="nfl_bet_log.csv",
                mime="text/csv",
                key="bet_dl",
            )
            up = st.file_uploader("Import bet log CSV", type=["csv"], key="bet_up")
            if up is not None:
                try:
                    imported = pd.read_csv(up)
                    _save_bet_log(imported)
                    st.success("Imported.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    elif adv == "Backtest":
        st.markdown("##### Simple Backtest")
        min_edge = st.slider("Minimum EPA edge", 0.03, 0.20, 0.05, 0.01)
        eval_seasons = st.multiselect("Evaluation seasons", [2021, 2022, 2023, 2024, 2025], default=[2023, 2024, 2025])
        train_seasons = st.multiselect("Train seasons", [2019, 2020, 2021, 2022, 2023, 2024], default=[2020, 2021, 2022])
        if st.button("Run Backtest"):
            with st.spinner("Training & evaluating..."):
                try:
                    mb = train_ats_model(train_seasons)
                    if not mb:
                        st.error("Not enough historical data.")
                    else:
                        model, cols = mb
                        hist_sched = load_schedules(eval_seasons)
                        hist_epa = get_team_epa(eval_seasons)
                        completed = hist_sched[hist_sched["result"].notna() & hist_sched["spread_line"].notna()]
                        results = []
                        for _, row in completed.iterrows():
                            home = row["home_team"]
                            away = row["away_team"]
                            if home not in hist_epa.index or away not in hist_epa.index:
                                continue
                            epa_edge = (
                                (hist_epa.loc[home, "off_epa"] - hist_epa.loc[away, "def_epa"]) -
                                (hist_epa.loc[away, "off_epa"] - hist_epa.loc[home, "def_epa"])
                            )
                            spread = float(row["spread_line"])
                            result = float(row["result"])
                            total_line = row.get("total_line", 45.0)
                            if pd.isna(total_line):
                                total_line = 45.0
                            feat = pd.DataFrame([{
                                "epa_edge": epa_edge, "spread": spread, "rest_diff": 0.0,
                                "home_off": hist_epa.loc[home, "off_epa"],
                                "home_def": hist_epa.loc[home, "def_epa"],
                                "away_off": hist_epa.loc[away, "off_epa"],
                                "away_def": hist_epa.loc[away, "def_epa"],
                                "abs_spread": abs(spread), "total_line": float(total_line)
                            }])[cols]
                            ml_prob = float(model.predict_proba(feat)[0, 1])
                            if epa_edge >= min_edge and ml_prob > 0.52:
                                results.append({"side": "Home", "covered": result > spread})
                            elif epa_edge <= -min_edge and ml_prob < 0.48:
                                results.append({"side": "Away", "covered": result < spread})
                        if results:
                            res_df = pd.DataFrame(results)
                            st.metric("ATS Win Rate", f"{res_df['covered'].mean():.1%}", delta=f"{len(res_df)} bets")
                        else:
                            st.warning("No games met the filters.")
                except Exception as e:
                    st.error(f"Backtest error: {e}")



    

# ---- Footer ----
st.markdown(
    """
<div class="nsc-footer">
  Research only — not betting advice. Data sources: nflverse / nflreadpy, The Odds API, Open-Meteo, NFL.com injuries, Ourlads depth charts, ESPN schedule.
</div>
    """,
    unsafe_allow_html=True,
)

