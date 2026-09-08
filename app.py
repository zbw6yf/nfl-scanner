import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import nflreadpy as nfl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NFL Opportunity Scanner – ML + Monte Carlo + Weather",
    page_icon="🏈",
    layout="wide"
)
st.title("🏈 NFL Betting Opportunity Scanner")
st.caption("EPA + Rules + Rest + Weather + ML + Monte Carlo. Research tool only. Not financial advice.")

# -----------------------------
# TEAM NAME MAPPING
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

def to_abbr(name: str) -> str | None:
    if not name:
        return None
    name = name.strip()
    if name in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[name]
    if len(name) <= 3 and name.isupper():
        return name
    return None

# -----------------------------
# STADIUM COORDINATES
# -----------------------------
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
n_simulations = st.sidebar.slider("Monte Carlo simulations", 2000, 15000, 8000, 1000)
st.sidebar.info("Player Props usually require a higher paid plan on The Odds API.")
st.sidebar.caption("Weather is fetched once per page load and cached.")

# -----------------------------
# DATA LOADERS (cached)
# -----------------------------
@st.cache_data(ttl=3600 * 6, show_spinner=False)
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
        if pbp.empty:
            return pd.DataFrame()
        off = pbp.groupby("posteam")["epa"].agg(off_epa="mean").reset_index().rename(columns={"posteam": "team"})
        deff = pbp.groupby("defteam")["epa"].agg(def_epa="mean").reset_index().rename(columns={"defteam": "team"})
        return off.merge(deff, on="team", how="outer").set_index("team")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_schedules(seasons=None):
    try:
        if seasons is None:
            current = nfl.get_current_season()
            seasons = list(range(current - 3, current + 1))
        sched = nfl.load_schedules(seasons=seasons)
        return sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_nfl_odds(api_key: str):
    if not api_key:
        return None, "No API key provided"
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            remaining = r.headers.get("x-requests-remaining", "?")
            return data, f"OK – {len(data)} events (remaining: {remaining})"
        return None, f"API error {r.status_code}: {r.text[:250]}"
    except Exception as e:
        return None, f"Request failed: {e}"

def fetch_player_props(api_key: str, event_id: str):
    if not api_key or not event_id:
        return None
    markets = ",".join([
        "player_pass_yds", "player_pass_tds", "player_rush_yds",
        "player_reception_yds", "player_receptions", "player_anytime_td",
        "player_pass_completions"
    ])
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {"apiKey": api_key, "regions": "us", "markets": markets, "oddsFormat": "american"}
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
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

def get_roof_for_game(schedules, home_abbr, game_date):
    try:
        mask = (
            (schedules["home_team"] == home_abbr) &
            (schedules["gameday"].astype(str).str[:10] == str(game_date)[:10])
        )
        rows = schedules[mask]
        if not rows.empty and "roof" in rows.columns:
            roof = rows.iloc[0]["roof"]
            if pd.notna(roof):
                return str(roof).lower()
        home_rows = schedules[schedules["home_team"] == home_abbr].dropna(subset=["roof"])
        if not home_rows.empty:
            return str(home_rows.iloc[-1]["roof"]).lower()
    except Exception:
        pass
    return "outdoors"

# -----------------------------
# WEATHER – fetched once per page load
# -----------------------------
@st.cache_data(ttl=3600 * 2, show_spinner=False)   # cache for 2 hours
def _fetch_single_weather(lat: float, lon: float, kickoff_iso: str):
    """Low-level Open-Meteo call. Never call this directly in a loop."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
            "forecast_days": 10
        }
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return {"temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0, "source": "fallback"}
        data = r.json()
        times = data.get("hourly", {}).get("time", [])
        temps = data.get("hourly", {}).get("temperature_2m", [])
        winds = data.get("hourly", {}).get("wind_speed_10m", [])
        precs = data.get("hourly", {}).get("precipitation_probability", [])
        if not times:
            return {"temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0, "source": "fallback"}

        kick = pd.to_datetime(kickoff_iso)
        if kick.tzinfo is None:
            kick = kick.tz_localize("UTC")
        best_idx, best_diff = 0, float("inf")
        for i, t in enumerate(times):
            tt = pd.to_datetime(t)
            if tt.tzinfo is None:
                tt = tt.tz_localize("UTC")
            diff = abs((tt - kick).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        return {
            "temp_f": float(temps[best_idx]) if best_idx < len(temps) else 70.0,
            "wind_mph": float(winds[best_idx]) if best_idx < len(winds) else 5.0,
            "precip_prob": float(precs[best_idx]) if best_idx < len(precs) else 10.0,
            "source": "open-meteo"
        }
    except Exception:
        return {"temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0, "source": "fallback"}

def preload_weather_for_games(odds_data, schedules):
    """
    Build a dictionary of weather for every outdoor game.
    This runs only once per page load (thanks to session_state).
    """
    if "weather_cache" in st.session_state and st.session_state.get("weather_cache_ready"):
        return st.session_state["weather_cache"]

    weather_dict = {}
    needed = []

    for game in odds_data or []:
        home_full = game.get("home_team")
        home = to_abbr(home_full)
        if not home or home not in STADIUM_COORDS:
            continue
        commence_raw = game.get("commence_time", "")
        game_date = commence_raw[:10] if commence_raw else datetime.now().strftime("%Y-%m-%d")
        roof = get_roof_for_game(schedules, home, game_date)
        if roof in ("dome", "closed"):
            weather_dict[home] = {
                "temp_f": 72.0, "wind_mph": 0.0, "precip_prob": 0.0,
                "source": "dome", "roof": roof
            }
            continue
        # key by team + date hour so we don't re-fetch the same stadium multiple times
        key = f"{home}_{commence_raw[:13]}"  # e.g. KC_2026-09-08T20
        needed.append((key, home, STADIUM_COORDS[home], commence_raw or f"{game_date}T17:00:00Z", roof))

    # Fetch only unique locations
    for key, home, (lat, lon), kickoff, roof in needed:
        if key not in weather_dict:
            wx = _fetch_single_weather(lat, lon, kickoff)
            wx["roof"] = roof
            weather_dict[key] = wx
            # also store under plain team key for easy lookup
            weather_dict[home] = wx

    st.session_state["weather_cache"] = weather_dict
    st.session_state["weather_cache_ready"] = True
    return weather_dict

def weather_adjustments(roof: str, weather: dict):
    is_indoor = roof in ("dome", "closed")
    if is_indoor:
        return {
            "total_adj": 0.0, "noise_extra": 0.0, "under_bias": 0.0,
            "rule_pts": 0.0, "label": "Dome / Closed (neutral)"
        }

    temp = weather.get("temp_f", 70)
    wind = weather.get("wind_mph", 5)
    precip = weather.get("precip_prob", 10)

    total_adj = noise_extra = under_bias = rule_pts = 0.0
    labels = []

    if wind >= 20:
        total_adj -= 3.5; noise_extra += 2.5; under_bias += 0.04; rule_pts += 1.4
        labels.append(f"High wind {wind:.0f} mph")
    elif wind >= 15:
        total_adj -= 2.0; noise_extra += 1.5; under_bias += 0.025; rule_pts += 0.9
        labels.append(f"Elevated wind {wind:.0f} mph")

    if precip >= 60:
        total_adj -= 2.5; noise_extra += 2.0; under_bias += 0.03; rule_pts += 1.1
        labels.append(f"High precip {precip:.0f}%")
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
        labels.append(f"Very hot {temp:.0f}°F")

    label = " • ".join(labels) if labels else f"Outdoor ({temp:.0f}°F / {wind:.0f} mph)"
    return {
        "total_adj": total_adj, "noise_extra": noise_extra,
        "under_bias": under_bias, "rule_pts": rule_pts, "label": label
    }

# -----------------------------
# ML + MONTE CARLO
# -----------------------------
@st.cache_data(ttl=3600 * 12, show_spinner=False)
def prepare_historical_features(seasons):
    try:
        sched = load_schedules(seasons=seasons)
        epa = get_team_epa(seasons=seasons)
        if sched.empty or epa.empty:
            return None, None
        completed = sched[
            sched["result"].notna() & sched["spread_line"].notna() &
            sched["home_score"].notna() & sched["away_score"].notna()
        ].copy()
        rows = []
        for _, row in completed.iterrows():
            home, away = row["home_team"], row["away_team"]
            if home not in epa.index or away not in epa.index:
                continue
            home_off = epa.loc[home, "off_epa"]
            home_def = epa.loc[home, "def_epa"]
            away_off = epa.loc[away, "off_epa"]
            away_def = epa.loc[away, "def_epa"]
            epa_edge = (home_off - away_def) - (away_off - home_def)
            spread = float(row["spread_line"])
            result = float(row["result"])
            covered = 1 if result > spread else 0
            rest_diff = 0
            total_line = row.get("total_line", 45.0)
            if pd.isna(total_line):
                total_line = 45.0
            rows.append({
                "epa_edge": epa_edge, "spread": spread, "rest_diff": rest_diff,
                "home_off": home_off, "home_def": home_def,
                "away_off": away_off, "away_def": away_def,
                "abs_spread": abs(spread), "total_line": total_line,
                "home_covered": covered
            })
        df = pd.DataFrame(rows)
        if len(df) < 80:
            return None, None
        return df, epa
    except Exception:
        return None, None

@st.cache_resource(ttl=3600 * 12, show_spinner=False)
def train_ats_model(seasons):
    hist, _ = prepare_historical_features(seasons)
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
    noise_std=11.5, total_adj=0.0, noise_extra=0.0, under_bias=0.0
):
    expected_margin = (home_off - away_def - (away_off - home_def)) * 35.0 + 1.2
    sim_margins = np.random.normal(loc=expected_margin, scale=noise_std + noise_extra, size=n_sims)
    expected_total = 44.0 + (home_off + away_off - home_def - away_def) * 22.0 + total_adj
    sim_totals = np.random.normal(loc=expected_total, scale=13.5 + noise_extra * 0.8, size=n_sims)

    home_cover_prob = float(np.mean(sim_margins > spread))
    over_prob = float(np.mean(sim_totals > total_line)) if total_line else 0.5
    over_prob = max(0.05, min(0.95, over_prob - under_bias))
    under_prob = 1.0 - over_prob

    home_ev = home_cover_prob * 100 / 110 - (1 - home_cover_prob)
    away_ev = (1 - home_cover_prob) * 100 / 110 - home_cover_prob

    return {
        "home_cover_prob": home_cover_prob,
        "away_cover_prob": 1.0 - home_cover_prob,
        "over_prob": over_prob,
        "under_prob": under_prob,
        "home_ev": float(home_ev),
        "away_ev": float(away_ev),
    }

# -----------------------------
# MAIN TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Opportunities",
    "📅 Games & Odds",
    "🎯 Player Props",
    "📊 Backtest",
    "🚀 Deploy"
])

with tab1:
    st.subheader("Ranked Game Opportunities – EPA + Weather + ML + Monte Carlo")

    # ---- Load everything once ----
    with st.spinner("Loading EPA, schedules, odds & weather (this runs only once per page load)..."):
        team_epa = get_team_epa()
        schedules = load_schedules()
        odds_data, odds_status = fetch_nfl_odds(api_key) if api_key else (None, "No API key")
        current_season = nfl.get_current_season()
        model_bundle = train_ats_model(list(range(current_season - 4, current_season)))

        # Pre-load weather exactly once
        weather_cache = {}
        if odds_data:
            weather_cache = preload_weather_for_games(odds_data, schedules)

    # Diagnostics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EPA teams", len(team_epa) if not team_epa.empty else 0)
    c2.metric("Odds events", len(odds_data) if odds_data else 0)
    c3.metric("Model", "Ready" if model_bundle else "No")
    c4.metric("Weather cached", len([k for k in weather_cache if "_" not in str(k)]))
    st.caption(f"Odds status: {odds_status}")

    opportunities = []
    skipped = []

    if odds_data and not team_epa.empty:
        model = model_bundle[0] if model_bundle else None
        feature_cols = model_bundle[1] if model_bundle else None

        for game in odds_data:
            home_full = game.get("home_team")
            away_full = game.get("away_team")
            home = to_abbr(home_full)
            away = to_abbr(away_full)

            if not home or not away:
                skipped.append(f"Unmapped: {away_full} @ {home_full}")
                continue
            if home not in team_epa.index or away not in team_epa.index:
                skipped.append(f"No EPA: {away} @ {home}")
                continue

            commence_raw = game.get("commence_time", "")
            commence = commence_raw[:16].replace("T", " ") if commence_raw else ""
            game_date = commence[:10] if commence else datetime.now().strftime("%Y-%m-%d")

            # Roof + Weather (already pre-fetched)
            roof = get_roof_for_game(schedules, home, game_date)
            wx_key = f"{home}_{commence_raw[:13]}" if commence_raw else home
            weather = weather_cache.get(wx_key) or weather_cache.get(home) or {
                "temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0, "source": "fallback", "roof": roof
            }
            wx_adj = weather_adjustments(roof, weather)

            # Lines
            spreads, totals = [], []
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market["key"] == "spreads":
                        for o in market["outcomes"]:
                            if o["name"] == home_full:
                                spreads.append(o.get("point"))
                    elif market["key"] == "totals":
                        for o in market["outcomes"]:
                            if o["name"] == "Over":
                                totals.append(o.get("point"))
            avg_spread = float(np.mean(spreads)) if spreads else None
            avg_total = float(np.mean(totals)) if totals else 45.0

            home_off = team_epa.loc[home, "off_epa"]
            home_def = team_epa.loc[home, "def_epa"]
            away_off = team_epa.loc[away, "off_epa"]
            away_def = team_epa.loc[away, "def_epa"]
            epa_edge = (home_off - away_def) - (away_off - home_def)

            home_rest = get_rest_days(schedules, home, game_date)
            away_rest = get_rest_days(schedules, away, game_date)
            rest_diff = home_rest - away_rest

            # Rules
            signals, rule_score = [], 0.0
            if epa_edge > 0.08:
                signals.append(f"Home EPA edge (+{epa_edge:.3f})"); rule_score += 2.2
            elif epa_edge < -0.08:
                signals.append(f"Away EPA edge ({epa_edge:.3f})"); rule_score += 2.0
            if avg_spread is not None and avg_spread > 1.5:
                signals.append("Home underdog"); rule_score += 1.3
            if avg_spread is not None and abs(avg_spread) >= 7:
                signals.append(f"Large spread ({avg_spread:+.1f})"); rule_score += 0.7
            if avg_total is not None and avg_total >= 48.5:
                signals.append(f"High total ({avg_total:.1f})"); rule_score += 0.6
            if rest_diff >= 3:
                signals.append(f"Home rest +{rest_diff}d"); rule_score += 1.1
            elif rest_diff <= -3:
                signals.append(f"Away rest {rest_diff}d"); rule_score += 1.0
            if wx_adj["rule_pts"] > 0:
                signals.append(wx_adj["label"]); rule_score += wx_adj["rule_pts"]

            # ML
            ml_home_cover = 0.5
            if model is not None and avg_spread is not None:
                feat = pd.DataFrame([{
                    "epa_edge": epa_edge, "spread": avg_spread, "rest_diff": rest_diff,
                    "home_off": home_off, "home_def": home_def,
                    "away_off": away_off, "away_def": away_def,
                    "abs_spread": abs(avg_spread), "total_line": avg_total
                }])[feature_cols]
                ml_home_cover = float(model.predict_proba(feat)[0, 1])

            # Monte Carlo
            mc = monte_carlo_game(
                home_off, home_def, away_off, away_def,
                avg_spread if avg_spread is not None else 0.0,
                avg_total, n_sims=n_simulations,
                total_adj=wx_adj["total_adj"],
                noise_extra=wx_adj["noise_extra"],
                under_bias=wx_adj["under_bias"]
            )

            ml_edge = abs(ml_home_cover - 0.5) * 4.0
            mc_edge = max(mc["home_ev"], mc["away_ev"]) * 8.0
            preferred = "Home" if (ml_home_cover > 0.5 and mc["home_cover_prob"] > 0.52) or \
                                 (ml_home_cover < 0.5 and mc["home_cover_prob"] < 0.48) else "Split"
            total_score = rule_score + ml_edge + mc_edge + (1.5 if preferred != "Split" else 0)

            if mc["home_ev"] > 0.03 and ml_home_cover > 0.53:
                rec = "Lean Home ATS"
            elif mc["away_ev"] > 0.03 and ml_home_cover < 0.47:
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
                wx_str = f"{weather.get('temp_f', 70):.0f}°F / {weather.get('wind_mph', 5):.0f} mph / {weather.get('precip_prob', 10):.0f}%"

            if signals or total_score > 2.0:
                opportunities.append({
                    "Game": f"{away_full} @ {home_full}",
                    "Kickoff": commence,
                    "Roof": roof.title(),
                    "Weather": wx_str,
                    "Spread": f"{avg_spread:+.1f}" if avg_spread is not None else "—",
                    "Total": f"{avg_total:.1f}" if avg_total is not None else "—",
                    "EPA Edge": f"{epa_edge:+.3f}",
                    "ML Home %": f"{ml_home_cover*100:.1f}%",
                    "MC Home %": f"{mc['home_cover_prob']*100:.1f}%",
                    "MC Over %": f"{mc['over_prob']*100:.1f}%",
                    "Recommendation": rec,
                    "Signals": " • ".join(signals) if signals else "—",
                    "Score": round(total_score, 2)
                })

        if opportunities:
            df = pd.DataFrame(opportunities).sort_values("Score", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("""
            **Weather handling**  
            - Dome / Closed → neutral  
            - Outdoor → Open-Meteo forecast fetched **once** at page load and cached  
            - High wind / precip / extreme temp adjust totals, variance, and rule score  
            """)
        else:
            st.warning("No opportunities found under current filters.")
            if skipped:
                with st.expander("Skipped"):
                    st.write(skipped)
    else:
        if not api_key:
            st.info("Enter your The Odds API key in the sidebar.")
        elif not odds_data:
            st.error(f"Odds error: {odds_status}")
        else:
            st.error("EPA data could not be loaded.")

# ========== TAB 2 ==========
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
        st.info("Enter API key to load games." if not api_key else odds_status)

# ========== TAB 3 ==========
with tab3:
    st.subheader("Player Props")
    if not api_key:
        st.warning("Enter API key first.")
    elif not odds_data:
        st.info("No games available.")
    else:
        game_options = {f"{g['away_team']} @ {g['home_team']}": g["id"] for g in odds_data}
        selected = st.selectbox("Game", list(game_options.keys()))
        if st.button("Load Props", type="primary"):
            with st.spinner("Fetching props..."):
                props = fetch_player_props(api_key, game_options[selected])
            if not props or "error" in (props or {}):
                st.error(props.get("error", "Failed") if props else "Failed")
            else:
                rows = []
                for book in props.get("bookmakers", []):
                    for market in book.get("markets", []):
                        for o in market.get("outcomes", []):
                            rows.append({
                                "Book": book.get("title"),
                                "Market": market.get("key", "").replace("player_", "").replace("_", " ").title(),
                                "Player": o.get("description", o.get("name")),
                                "Side": o.get("name"),
                                "Line": o.get("point"),
                                "Odds": o.get("price")
                            })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.warning("No props returned (plan limitation).")

# ========== TAB 4 ==========
with tab4:
    st.subheader("Backtest")
    min_edge = st.slider("Min EPA edge", 0.03, 0.20, 0.05, 0.01)
    seasons_back = st.multiselect("Eval seasons", [2021, 2022, 2023, 2024, 2025], default=[2023, 2024, 2025])
    train_on = st.multiselect("Train seasons", [2019, 2020, 2021, 2022, 2023, 2024], default=[2020, 2021, 2022])
    if st.button("Run Backtest"):
        with st.spinner("Running..."):
            try:
                mb = train_ats_model(train_on)
                if not mb:
                    st.error("Not enough data.")
                else:
                    model, cols = mb
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
                        spread = float(row["spread_line"])
                        result = float(row["result"])
                        total_line = row.get("total_line", 45.0) or 45.0
                        feat = pd.DataFrame([{
                            "epa_edge": epa_edge, "spread": spread, "rest_diff": 0,
                            "home_off": hist_epa.loc[home, "off_epa"],
                            "home_def": hist_epa.loc[home, "def_epa"],
                            "away_off": hist_epa.loc[away, "off_epa"],
                            "away_def": hist_epa.loc[away, "def_epa"],
                            "abs_spread": abs(spread), "total_line": total_line
                        }])[cols]
                        ml_prob = float(model.predict_proba(feat)[0, 1])
                        if epa_edge >= min_edge and ml_prob > 0.52:
                            results.append({"side": "Home", "covered": result > spread})
                        elif epa_edge <= -min_edge and ml_prob < 0.48:
                            results.append({"side": "Away", "covered": result < spread})
                    if results:
                        st.metric("ATS Win Rate", f"{pd.DataFrame(results)['covered'].mean():.1%}",
                                  delta=f"{len(results)} bets")
                    else:
                        st.warning("No qualifying games.")
            except Exception as e:
                st.error(str(e))

with tab5:
    st.write("Deploy notes / Streamlit Cloud tips.")
