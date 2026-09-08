import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
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


# -----------------------------
# Embedded 2026 REG schedule Weeks 1-10 (ESPN official) — source of truth for matchups/dates/times
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
]


# BUILD MASTER GAME LIST FROM SCHEDULE (source of truth for weeks / dates)
# -----------------------------
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
            week = get_week(schedules, home, away, game_date) if (schedules is not None and not schedules.empty) else estimate_week_from_date(game_date)
        roof = get_roof(schedules, home, game_date) if (schedules is not None and not schedules.empty) else "outdoors"
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
    cutoff = today + pd.Timedelta(days=max(days_ahead, 120))

    # ---- Primary: embedded official slate ----
    for row in EMBEDDED_2026_SCHEDULE:
        try:
            gameday = row["gameday"]
            gd = pd.to_datetime(gameday, errors="coerce")
            if pd.isna(gd):
                continue
            # Keep games from 2 days ago through cutoff (full weeks intact)
            if gd < today - pd.Timedelta(days=2) or gd > cutoff:
                continue
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
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Opportunities",
    "📅 Games & Odds",
    "🎯 Player Props",
    "📊 Backtest"
])
# ========== TAB 1 ==========
with tab1:
    st.subheader("Ranked Opportunities")
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
        weather_cache = build_weather_cache_from_games(upcoming)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("EPA teams", 0 if team_epa.empty else len(team_epa))
    c2.metric("Upcoming games", len(upcoming))
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
    st.caption(odds_status + f" · {len(upcoming)} upcoming games loaded")
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
    if upcoming and not team_epa.empty:
        model = model_bundle[0] if model_bundle else None
        feature_cols = model_bundle[1] if model_bundle else None
        league_avg_pace = float(team_pace["plays_per_game"].mean()) if not team_pace.empty else 65.0
        for g in upcoming:
            try:
                home = g["home"]
                away = g["away"]
                home_full = g["home_full"]
                away_full = g["away_full"]
                if home not in team_epa.index or away not in team_epa.index:
                    skipped.append(f"No EPA for {away} @ {home}")
                    continue
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

                home_off = float(team_epa.loc[home, "off_epa"])
                home_def = float(team_epa.loc[home, "def_epa"])
                away_off = float(team_epa.loc[away, "off_epa"])
                away_def = float(team_epa.loc[away, "def_epa"])
                epa_edge = (home_off - away_def) - (away_off - home_def)
                rest_diff = get_rest_days(schedules, home, game_date) - get_rest_days(schedules, away, game_date)
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
                if roof in ("dome", "closed"):
                    wx_str = "Dome"
                else:
                    wx_str = (f"{weather.get('temp_f', 70):.0f}°F / "
                              f"{weather.get('wind_mph', 5):.0f} mph / "
                              f"{weather.get('precip_prob', 10):.0f}%")
                week_num = g.get("week")
                if week_num is None:
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

            # ---- FILTER CONTROLS ----
            st.markdown("##### Filters")
            f1, f2, f3 = st.columns([1, 1.4, 1])
            with f1:
                min_score = st.slider(
                    "Min Score",
                    min_value=0.0,
                    max_value=max(10.0, float(df["Score"].max()) if len(df) else 10.0),
                    value=2.0,
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
                    default=["Home ATS", "Away ATS", "Over", "Under"],
                    key="opp_rec_filter",
                )
            with f3:
                df["_Week_num"] = pd.to_numeric(df["Week"], errors="coerce")
                available_weeks = sorted(df["_Week_num"].dropna().unique().tolist())
                week_choices = ["All weeks"] + [f"Week {int(w)}" for w in available_weeks]
                selected_week_filter = st.selectbox(
                    "Week",
                    options=week_choices,
                    index=0,
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

            st.caption(
                f"Showing **{len(filtered)}** of **{len(df)}** opportunities "
                f"(Min Score ≥ {min_score}"
                + (f", Week filter: {selected_week_filter}" if selected_week_filter != "All weeks" else "")
                + ")"
            )
            display_df = filtered.drop(columns=["_Week_num"], errors="ignore")
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
                "EPA Edge", "Form Δ", "Recommendation", "Score", "Signals"
            ]
            if not df_known.empty:
                weeks_sorted = sorted(df_known["Week_num"].unique())
                week_labels = {int(w): f"Week {int(w)}" for w in weeks_sorted}
                default_idx = 0
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
    cutoff = today + pd.Timedelta(days=120)
    rows = []
    for row in EMBEDDED_2026_SCHEDULE:
        try:
            gameday = row["gameday"]
            gd = pd.to_datetime(gameday, errors="coerce")
            if pd.isna(gd) or gd < today - pd.Timedelta(days=2) or gd > cutoff:
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
        weeks_present = sorted(games_df["Week"].dropna().unique().tolist())
        week_filter = st.selectbox(
            "Filter by week",
            options=["All weeks"] + [f"Week {int(w)}" for w in weeks_present],
            key="tab2_week_filter",
        )
        display = games_df
        if week_filter != "All weeks":
            try:
                wk = int(week_filter.replace("Week ", ""))
                display = games_df[games_df["Week"] == wk]
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

