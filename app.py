import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime
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
    layout="wide"
)
st.title("🏈 NFL Betting Opportunity Scanner")
st.caption("EPA + Rules + Rest + Weather + ML + Monte Carlo + Implied Totals + Form + Pace + Travel + Divisional · Research tool only")
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
api_key = st.sidebar.text_input("The Odds API Key", type="password")
n_simulations = st.sidebar.slider("Monte Carlo simulations", 2000, 15000, 8000, 1000)
form_window = st.sidebar.slider("Recent form window (games)", 4, 8, 6, 1)
if st.sidebar.button("Clear all caches"):
    st.cache_data.clear()
    st.cache_resource.clear()
    for k in list(st.session_state.keys()):
        if "weather" in k.lower():
            del st.session_state[k]
    st.rerun()
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
        # Count offensive plays per team per game
        plays = pbp[
            (pbp["play_type"].isin(["pass", "run"])) &
            (pbp["posteam"].notna())
        ].copy()
        if plays.empty:
            return pd.DataFrame()
        # Group by game_id + posteam
        g = plays.groupby(["game_id", "posteam"]).size().reset_index(name="off_plays")
        # Average plays per game (offense)
        pace = g.groupby("posteam")["off_plays"].mean().reset_index()
        pace.columns = ["team", "plays_per_game"]
        return pace.set_index("team")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_recent_form(seasons: Optional[List[int]] = None, n_games: int = 6) -> Dict[str, Dict]:
    """
    Last N completed games: average EPA (off - def) and average margin (points).
    Returns dict[team] = {"form_epa": float, "form_margin": float, "n": int}
    """
    try:
        if seasons is None:
            current = int(nfl.get_current_season())
            seasons = [current - 1, current]
        sched = nfl.load_schedules(seasons=seasons)
        if hasattr(sched, "to_pandas"):
            sched = sched.to_pandas()
        if sched is None or sched.empty:
            return {}
        # Completed games only
        completed = sched[
            sched["result"].notna() &
            sched["home_score"].notna() &
            sched["away_score"].notna()
        ].copy()
        if completed.empty:
            return {}
        completed["gameday"] = pd.to_datetime(completed["gameday"])
        completed = completed.sort_values("gameday")
        # Also need EPA if possible – fallback to margin only if EPA unavailable
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
                # Approximate recent EPA differential if available
                if not epa_df.empty and team in epa_df.index:
                    opp = row["away_team"] if row["home_team"] == team else row["home_team"]
                    if opp in epa_df.index:
                        # crude: team's off EPA vs opp def EPA
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

@st.cache_data(ttl=3600, show_spinner=False)
def load_schedules(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    try:
        if seasons is None:
            current = int(nfl.get_current_season())
            seasons = list(range(current - 3, current + 1))
        sched = nfl.load_schedules(seasons=seasons)
        if hasattr(sched, "to_pandas"):
            sched = sched.to_pandas()
        return sched if isinstance(sched, pd.DataFrame) else pd.DataFrame()
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

def get_rest_days(schedules: pd.DataFrame, team: str, game_date: str) -> int:
    try:
        if schedules.empty or "home_team" not in schedules.columns:
            return 7
        mask = (
            ((schedules["home_team"] == team) | (schedules["away_team"] == team)) &
            (schedules["gameday"].astype(str) < str(game_date)[:10])
        )
        prior = schedules.loc[mask].sort_values("gameday")
        if prior.empty:
            return 7
        last = str(prior.iloc[-1]["gameday"])
        return max((pd.to_datetime(game_date[:10]) - pd.to_datetime(last[:10])).days, 0)
    except Exception:
        return 7

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
    Week 1 ≈ first Thursday on/after Sept 4, with a few days of buffer for Wed openers.
    """
    try:
        target = pd.to_datetime(str(game_date)[:10], errors="coerce")
        if pd.isna(target):
            return None
        year = target.year if target.month >= 3 else target.year - 1
        start = pd.Timestamp(year=year, month=9, day=4)
        while start.weekday() != 3:  # Thursday
            start += pd.Timedelta(days=1)
        week1_start = start - pd.Timedelta(days=3)
        if target < week1_start - pd.Timedelta(days=7):
            return None
        days_since = (target - week1_start).days
        if days_since < 0:
            return 1
        week = days_since // 7 + 1
        if week < 1:
            return 1
        if week > 22:
            return None
        return int(week)
    except Exception:
        return None

def get_week(schedules: pd.DataFrame, home: str, away: str, game_date: str) -> Optional[int]:
    """
    Look up NFL week from schedules (current season, abbr aliases, tight dates).
    Falls back to calendar estimate so Top-5-by-Week never goes blank.
    """
    try:
        gd = str(game_date)[:10]
        target = pd.to_datetime(gd, errors="coerce")
        if pd.isna(target):
            return estimate_week_from_date(game_date)

        try:
            current = int(nfl.get_current_season())
        except Exception:
            current = target.year if target.month >= 8 else target.year - 1

        sched = schedules.copy() if schedules is not None and not getattr(schedules, "empty", True) else pd.DataFrame()
        if not sched.empty and "season" in sched.columns:
            sched_cur = sched[sched["season"] == current]
            if not sched_cur.empty:
                sched = sched_cur

        if not sched.empty and "week" in sched.columns and "home_team" in sched.columns:
            home_set = _expand_team(home)
            away_set = _expand_team(away)

            def _week_if_close(rows: pd.DataFrame, max_days: int = 3) -> Optional[int]:
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
            week = _week_if_close(sched.loc[mask], max_days=3)
            if week is not None:
                return week

            mask_flip = sched["home_team"].isin(away_set) & sched["away_team"].isin(home_set)
            week = _week_if_close(sched.loc[mask_flip], max_days=3)
            if week is not None:
                return week

            mask2 = sched["home_team"].isin(home_set)
            week = _week_if_close(sched.loc[mask2], max_days=1)
            if week is not None:
                return week

            mask3 = sched["gameday"].astype(str).str[:10] == gd
            rows3 = sched.loc[mask3]
            if not rows3.empty and pd.notna(rows3.iloc[0].get("week")):
                return int(rows3.iloc[0]["week"])

        return estimate_week_from_date(game_date)
    except Exception:
        return estimate_week_from_date(game_date)

def implied_team_totals(spread: float, total: float) -> Tuple[float, float]:
    """
    spread = home team line (negative if home favorite).
    Returns (home_implied, away_implied).
    """
    # Standard: home_implied = (total - spread) / 2
    # away_implied = (total + spread) / 2
    home_imp = (total - spread) / 2.0
    away_imp = (total + spread) / 2.0
    return home_imp, away_imp
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

def build_weather_cache(odds_data: List, schedules: pd.DataFrame) -> Dict[str, Dict]:
    cache = {}
    real_count = 0
    fallback_count = 0
    samples = []
    if not odds_data:
        return cache
    progress = st.progress(0, text="Fetching unique weather for each outdoor stadium...")
    total = len(odds_data)
    for idx, game in enumerate(odds_data):
        try:
            home_full = game.get("home_team")
            home = to_abbr(home_full)
            if not home or home not in STADIUM_COORDS:
                continue
            commence_raw = game.get("commence_time") or ""
            game_date = commence_raw[:10] if len(commence_raw) >= 10 else datetime.now().strftime("%Y-%m-%d")
            roof = get_roof(schedules, home, game_date)
            key = make_weather_key(home, commence_raw)
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
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Opportunities",
    "📅 Games & Odds",
    "🎯 Player Props",
    "📊 Backtest"
])
# ========== TAB 1 ==========
with tab1:
    st.subheader("Ranked Opportunities")
    with st.spinner("Loading EPA, Pace, Form, Odds and unique weather..."):
        team_epa = get_team_epa()
        team_pace = get_team_pace()
        recent_form = get_recent_form(n_games=form_window)
        schedules = load_schedules()
        odds_data, odds_status = fetch_nfl_odds(api_key) if api_key else (None, "No API key entered")
        try:
            current_season = int(nfl.get_current_season())
        except Exception:
            current_season = 2025
        model_bundle = train_ats_model(list(range(current_season - 4, current_season)))
        weather_cache = build_weather_cache(odds_data or [], schedules)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("EPA teams", 0 if team_epa.empty else len(team_epa))
    c2.metric("Odds events", 0 if not odds_data else len(odds_data))
    c3.metric("Model", "Ready" if model_bundle else "Missing")
    c4.metric("Weather keys", len(weather_cache))
    c5.metric("Form teams", len(recent_form))
    debug = st.session_state.get("weather_debug", {})
    if debug:
        st.info(
            f"Weather → Real Open-Meteo: **{debug.get('real', 0)}** | "
            f"Fallbacks: **{debug.get('fallback', 0)}** | Keys: {debug.get('total_keys', 0)}"
        )
        if debug.get("samples"):
            st.caption("Sample real weather (should differ by stadium):")
            st.dataframe(pd.DataFrame(debug["samples"]), use_container_width=True, hide_index=True)
    st.caption(odds_status)
    opportunities = []
    skipped = []
    if odds_data and not team_epa.empty:
        model = model_bundle[0] if model_bundle else None
        feature_cols = model_bundle[1] if model_bundle else None
        league_avg_pace = float(team_pace["plays_per_game"].mean()) if not team_pace.empty else 65.0
        for game in odds_data:
            try:
                home_full = game.get("home_team", "")
                away_full = game.get("away_team", "")
                home = to_abbr(home_full)
                away = to_abbr(away_full)
                if not home or not away:
                    skipped.append(f"Unmapped: {away_full} @ {home_full}")
                    continue
                if home not in team_epa.index or away not in team_epa.index:
                    skipped.append(f"No EPA for {away} @ {home}")
                    continue
                commence_raw = game.get("commence_time") or ""
                commence = commence_raw[:16].replace("T", " ") if commence_raw else ""
                game_date = commence[:10] if commence else datetime.now().strftime("%Y-%m-%d")
                roof = get_roof(schedules, home, game_date)
                wx_key = make_weather_key(home, commence_raw)
                weather = weather_cache.get(wx_key) or {
                    "temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0,
                    "roof": roof, "source": "missing"
                }
                wx_adj = weather_adjustments(roof, weather)
                spreads, totals = [], []
                for book in game.get("bookmakers", []):
                    for market in book.get("markets", []):
                        if market.get("key") == "spreads":
                            for o in market.get("outcomes", []):
                                if o.get("name") == home_full:
                                    spreads.append(o.get("point"))
                        elif market.get("key") == "totals":
                            for o in market.get("outcomes", []):
                                if o.get("name") == "Over":
                                    totals.append(o.get("point"))
                avg_spread = float(np.mean(spreads)) if spreads else None
                avg_total = float(np.mean(totals)) if totals else 45.0
                home_off = float(team_epa.loc[home, "off_epa"])
                home_def = float(team_epa.loc[home, "def_epa"])
                away_off = float(team_epa.loc[away, "off_epa"])
                away_def = float(team_epa.loc[away, "def_epa"])
                epa_edge = (home_off - away_def) - (away_off - home_def)
                rest_diff = get_rest_days(schedules, home, game_date) - get_rest_days(schedules, away, game_date)
                # ---- NEW SIGNALS ----
                # 1. Implied Team Totals
                if avg_spread is not None:
                    home_imp, away_imp = implied_team_totals(avg_spread, avg_total)
                else:
                    home_imp, away_imp = avg_total / 2, avg_total / 2
                # 2. Recent Form
                home_form = recent_form.get(home, {"form_margin": 0.0, "form_epa": 0.0, "n": 0})
                away_form = recent_form.get(away, {"form_margin": 0.0, "form_epa": 0.0, "n": 0})
                form_margin_diff = home_form["form_margin"] - away_form["form_margin"]
                form_epa_diff = home_form["form_epa"] - away_form["form_epa"]
                # 3. Pace
                home_pace = float(team_pace.loc[home, "plays_per_game"]) if (not team_pace.empty and home in team_pace.index) else league_avg_pace
                away_pace = float(team_pace.loc[away, "plays_per_game"]) if (not team_pace.empty and away in team_pace.index) else league_avg_pace
                combined_pace = (home_pace + away_pace) / 2.0
                pace_vs_avg = combined_pace - league_avg_pace
                # Pace adjustment to total: ~0.4 pts per extra play above average
                pace_adj = pace_vs_avg * 0.35
                # 4. Travel / Time-zone
                tz_diff = timezone_diff(home, away)
                travel_dir = travel_direction(home, away)
                # 5. Divisional
                div_flag = is_divisional(home, away)
                signals = []
                rule_score = 0.0
                # Existing EPA / spread / rest / weather signals
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
                # NEW: Implied Team Totals signals (high value)
                if home_imp >= 27.5:
                    signals.append(f"High Home Imp {home_imp:.1f}"); rule_score += 1.5
                elif home_imp <= 17.5:
                    signals.append(f"Low Home Imp {home_imp:.1f}"); rule_score += 1.2
                if away_imp >= 27.5:
                    signals.append(f"High Away Imp {away_imp:.1f}"); rule_score += 1.4
                elif away_imp <= 17.5:
                    signals.append(f"Low Away Imp {away_imp:.1f}"); rule_score += 1.1
                # Implied total vs market total mismatch (value for totals)
                model_total_est = home_imp + away_imp  # should equal avg_total, but used for framing
                if avg_total >= 48 and (home_imp + away_imp) < 46:
                    signals.append("Implied soft total"); rule_score += 0.8
                # NEW: Recent Form
                if form_margin_diff >= 7:
                    signals.append(f"Home form +{form_margin_diff:.1f}"); rule_score += 1.6
                elif form_margin_diff <= -7:
                    signals.append(f"Away form {form_margin_diff:.1f}"); rule_score += 1.5
                if form_epa_diff > 0.12:
                    signals.append(f"Home form EPA +{form_epa_diff:.3f}"); rule_score += 1.3
                elif form_epa_diff < -0.12:
                    signals.append(f"Away form EPA {form_epa_diff:.3f}"); rule_score += 1.2
                # NEW: Pace
                if pace_vs_avg >= 4.0:
                    signals.append(f"Fast pace +{pace_vs_avg:.1f}"); rule_score += 1.0
                elif pace_vs_avg <= -4.0:
                    signals.append(f"Slow pace {pace_vs_avg:.1f}"); rule_score += 0.9
                # NEW: Travel / TZ
                if tz_diff >= 3:
                    if travel_dir == "Westbound":
                        signals.append(f"Away TZ -{tz_diff}h West"); rule_score += 1.1
                    else:
                        signals.append(f"Away TZ -{tz_diff}h East"); rule_score += 0.9
                elif tz_diff == 2:
                    signals.append(f"Away TZ -{tz_diff}h"); rule_score += 0.5
                # NEW: Divisional
                if div_flag:
                    signals.append("Divisional"); rule_score += 0.7
                # Form adjustment for Monte Carlo (scaled)
                form_margin_adj = form_margin_diff * 0.15  # soft contribution
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
                # Clearer weather display
                if roof in ("dome", "closed"):
                    wx_str = "Dome"
                else:
                    wx_str = (f"{weather.get('temp_f', 70):.0f}°F / "
                              f"{weather.get('wind_mph', 5):.0f} mph / "
                              f"{weather.get('precip_prob', 10):.0f}%")
                week_num = get_week(schedules, home, away, game_date)
                if signals or total_score > 2.0:
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
                        "ML Home %": f"{ml_home*100:.1f}%",
                        "MC Home %": f"{mc['home_cover_prob']*100:.1f}%",
                        "MC Over %": f"{mc['over_prob']*100:.1f}%",
                        "Recommendation": rec,
                        "Signals": " • ".join(signals) if signals else "—",
                        "Score": round(total_score, 2)
                    })
            except Exception as e:
                skipped.append(f"Error: {e}")
                continue
        if opportunities:
            df = pd.DataFrame(opportunities).sort_values("Score", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # ---- TOP 5 SIGNALED GAMES BY WEEK ----
            st.markdown("---")
            st.subheader("🏆 Top 5 Signaled Games by Week")
            st.caption(
                "Select a week from the dropdown to see its 5 highest-Score opportunities. "
                "Only weeks present in the current odds feed are listed."
            )
            df_week = df.copy()
            df_week["Week_num"] = pd.to_numeric(df_week["Week"], errors="coerce")
            df_known = df_week[df_week["Week_num"].notna()].copy()
            display_cols = [
                "Game", "Kickoff", "Spread", "Total", "Home Imp", "Away Imp",
                "EPA Edge", "Form Δ", "Recommendation", "Score", "Signals"
            ]
            if not df_known.empty:
                weeks_sorted = sorted(df_known["Week_num"].unique())
                week_labels = {int(w): f"Week {int(w)}" for w in weeks_sorted}
                # Default to the earliest upcoming week
                default_idx = 0
                selected_label = st.selectbox(
                    "Select week",
                    options=[week_labels[int(w)] for w in weeks_sorted],
                    index=default_idx,
                    key="top5_week_select",
                )
                # Map label back to week number
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

            # Quick summary of top signals
            st.markdown("#### Top Signal Summary")
            st.caption("Implied Team Totals, Recent Form, Pace, Travel/TZ and Divisional are now folded into Score + Signals.")
        else:
            st.warning("No opportunities matched the filters.")
            if skipped:
                with st.expander("Skipped"):
                    for s in skipped:
                        st.text(s)
    else:
        if not api_key:
            st.info("Enter your The Odds API key in the sidebar.")
        elif not odds_data:
            st.error(f"Could not load odds: {odds_status}")
        else:
            st.error("Could not load EPA data.")
# ========== TAB 2 ==========
with tab2:
    st.subheader("Upcoming Games")
    if odds_data:
        rows = []
        for g in odds_data:
            home = g.get("home_team", "")
            away = g.get("away_team", "")
            commence = (g.get("commence_time") or "")[:16].replace("T", " ")
            spread = total = "—"
            books = g.get("bookmakers") or []
            if books:
                for m in books[0].get("markets", []):
                    if m.get("key") == "spreads":
                        for o in m.get("outcomes", []):
                            if o.get("name") == home:
                                spread = f"{o.get('point', 0):+.1f}"
                    if m.get("key") == "totals":
                        for o in m.get("outcomes", []):
                            if o.get("name") == "Over":
                                total = f"{o.get('point', 0):.1f}"
            # Add implied + divisional for richer view
            home_a = to_abbr(home)
            away_a = to_abbr(away)
            imp_h = imp_a = "—"
            if spread != "—" and total != "—":
                try:
                    s = float(spread)
                    t = float(total)
                    ih, ia = implied_team_totals(s, t)
                    imp_h, imp_a = f"{ih:.1f}", f"{ia:.1f}"
                except Exception:
                    pass
            div = "Yes" if (home_a and away_a and is_divisional(home_a, away_a)) else "No"
            rows.append({
                "Away": away, "Home": home, "Kickoff": commence,
                "Spread": spread, "Total": total,
                "Home Imp": imp_h, "Away Imp": imp_a, "Divisional": div
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info(odds_status if api_key else "Enter API key")
# ========== TAB 3 ==========
with tab3:
    st.subheader("Player Props")
    if not api_key:
        st.warning("Enter API key first.")
    elif not odds_data:
        st.info("No games available.")
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
    st.subheader("Simple Backtest")
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
