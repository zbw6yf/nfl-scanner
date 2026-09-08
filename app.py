import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timezone
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
# TEAM NAME MAPPING (Odds API → nflverse abbr)
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
# STADIUM COORDINATES (lat, lon) for Open-Meteo
# -----------------------------
STADIUM_COORDS = {
    "ARI": (33.5275, -112.2625),   # State Farm Stadium
    "ATL": (33.7554, -84.4010),    # Mercedes-Benz Stadium
    "BAL": (39.2780, -76.6227),    # M&T Bank Stadium
    "BUF": (42.7738, -78.7870),    # Highmark Stadium
    "CAR": (35.2258, -80.8528),    # Bank of America Stadium
    "CHI": (41.8623, -87.6167),    # Soldier Field
    "CIN": (39.0950, -84.5160),    # Paycor Stadium
    "CLE": (41.5061, -81.6995),    # Cleveland Browns Stadium
    "DAL": (32.7473, -97.0945),    # AT&T Stadium
    "DEN": (39.7439, -105.0201),   # Empower Field
    "DET": (42.3400, -83.0456),    # Ford Field
    "GB":  (44.5013, -88.0622),    # Lambeau Field
    "HOU": (29.6847, -95.4107),    # NRG Stadium
    "IND": (39.7601, -86.1639),    # Lucas Oil Stadium
    "JAX": (30.3239, -81.6373),    # EverBank Stadium
    "KC":  (39.0489, -94.4839),    # Arrowhead
    "LAC": (33.9535, -118.3392),   # SoFi Stadium
    "LA":  (33.9535, -118.3392),   # SoFi Stadium
    "LV":  (36.0908, -115.1830),   # Allegiant Stadium
    "MIA": (25.9580, -80.2389),    # Hard Rock Stadium
    "MIN": (44.9738, -93.2581),    # U.S. Bank Stadium
    "NE":  (42.0909, -71.2643),    # Gillette Stadium
    "NO":  (29.9511, -90.0812),    # Caesars Superdome
    "NYG": (40.8128, -74.0742),    # MetLife
    "NYJ": (40.8128, -74.0742),    # MetLife
    "PHI": (39.9008, -75.1675),    # Lincoln Financial Field
    "PIT": (40.4468, -80.0158),    # Acrisure Stadium
    "SF":  (37.4033, -121.9694),   # Levi's Stadium
    "SEA": (47.5952, -122.3316),   # Lumen Field
    "TB":  (27.9759, -82.5033),    # Raymond James Stadium
    "TEN": (36.1665, -86.7713),    # Nissan Stadium
    "WAS": (38.9077, -76.8645),    # Northwest Stadium
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
st.sidebar.caption("Weather via Open-Meteo (free, no key). Dome games = neutral weather.")

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
        if pbp.empty:
            return pd.DataFrame()
        off = pbp.groupby("posteam")["epa"].agg(off_epa="mean").reset_index().rename(columns={"posteam": "team"})
        deff = pbp.groupby("defteam")["epa"].agg(def_epa="mean").reset_index().rename(columns={"defteam": "team"})
        team_epa = off.merge(deff, on="team", how="outer")
        return team_epa.set_index("team")
    except Exception as e:
        st.warning(f"EPA load issue: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_schedules(seasons=None):
    try:
        if seasons is None:
            current = nfl.get_current_season()
            seasons = list(range(current - 3, current + 1))
        sched = nfl.load_schedules(seasons=seasons)
        return sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    except Exception as e:
        st.warning(f"Schedules load issue: {e}")
        return pd.DataFrame()

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
        else:
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

def get_roof_for_game(schedules, home_abbr, game_date):
    """Return roof type for the home team's game on/near that date."""
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
        # fallback: most recent known roof for that home team
        home_rows = schedules[schedules["home_team"] == home_abbr].dropna(subset=["roof"])
        if not home_rows.empty:
            return str(home_rows.iloc[-1]["roof"]).lower()
    except Exception:
        pass
    return "outdoors"  # conservative default

@st.cache_data(ttl=1800)
def fetch_weather(lat: float, lon: float, kickoff_iso: str):
    """
    Fetch hourly forecast from Open-Meteo nearest to kickoff.
    Returns dict with temp_f, wind_mph, precip_prob, or neutral defaults on failure.
    """
    try:
        # Open-Meteo expects ISO time; we request hourly for the next ~10 days
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
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return {"temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0, "source": "fallback"}
        data = r.json()
        times = data.get("hourly", {}).get("time", [])
        temps = data.get("hourly", {}).get("temperature_2m", [])
        winds = data.get("hourly", {}).get("wind_speed_10m", [])
        precs = data.get("hourly", {}).get("precipitation_probability", [])
        if not times:
            return {"temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0, "source": "fallback"}

        # Find closest hour to kickoff
        kick = pd.to_datetime(kickoff_iso)
        if kick.tzinfo is None:
            kick = kick.tz_localize("UTC")
        best_idx = 0
        best_diff = float("inf")
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

def weather_adjustments(roof: str, weather: dict):
    """
    Return adjustments for expected total, margin noise, and rule score contribution.
    Dome / closed = neutral.
    """
    is_indoor = roof in ("dome", "closed")
    if is_indoor:
        return {
            "total_adj": 0.0,
            "noise_extra": 0.0,
            "under_bias": 0.0,
            "rule_pts": 0.0,
            "label": "Dome / Closed (neutral weather)"
        }

    temp = weather.get("temp_f", 70)
    wind = weather.get("wind_mph", 5)
    precip = weather.get("precip_prob", 10)

    total_adj = 0.0
    noise_extra = 0.0
    under_bias = 0.0
    rule_pts = 0.0
    labels = []

    if wind >= 20:
        total_adj -= 3.5
        noise_extra += 2.5
        under_bias += 0.04
        rule_pts += 1.4
        labels.append(f"High wind {wind:.0f} mph")
    elif wind >= 15:
        total_adj -= 2.0
        noise_extra += 1.5
        under_bias += 0.025
        rule_pts += 0.9
        labels.append(f"Elevated wind {wind:.0f} mph")

    if precip >= 60:
        total_adj -= 2.5
        noise_extra += 2.0
        under_bias += 0.03
        rule_pts += 1.1
        labels.append(f"High precip chance {precip:.0f}%")
    elif precip >= 40:
        total_adj -= 1.2
        noise_extra += 1.0
        under_bias += 0.015
        rule_pts += 0.6
        labels.append(f"Precip chance {precip:.0f}%")

    if temp <= 25:
        total_adj -= 2.0
        noise_extra += 1.5
        rule_pts += 0.7
        labels.append(f"Very cold {temp:.0f}°F")
    elif temp <= 35:
        total_adj -= 1.0
        noise_extra += 0.8
        rule_pts += 0.4
        labels.append(f"Cold {temp:.0f}°F")
    elif temp >= 95:
        total_adj -= 1.0
        noise_extra += 1.0
        rule_pts += 0.4
        labels.append(f"Very hot {temp:.0f}°F")

    label = " • ".join(labels) if labels else f"Outdoor ({temp:.0f}°F, wind {wind:.0f} mph)"
    return {
        "total_adj": total_adj,
        "noise_extra": noise_extra,
        "under_bias": under_bias,
        "rule_pts": rule_pts,
        "label": label
    }

# -----------------------------
# ML + MONTE CARLO
# -----------------------------
@st.cache_data(ttl=3600 * 12)
def prepare_historical_features(seasons):
    try:
        sched = load_schedules(seasons=seasons)
        epa = get_team_epa(seasons=seasons)
        if sched.empty or epa.empty:
            return None, None
        completed = sched[
            sched["result"].notna() &
            sched["spread_line"].notna() &
            sched["home_score"].notna() &
            sched["away_score"].notna()
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
            home_rest = 7
            away_rest = 7
            try:
                prior = sched[
                    ((sched["home_team"] == home) | (sched["away_team"] == home)) &
                    (sched["gameday"] < row["gameday"])
                ].sort_values("gameday")
                if not prior.empty:
                    home_rest = max((pd.to_datetime(row["gameday"]) - pd.to_datetime(prior.iloc[-1]["gameday"])).days, 0)
                prior = sched[
                    ((sched["home_team"] == away) | (sched["away_team"] == away)) &
                    (sched["gameday"] < row["gameday"])
                ].sort_values("gameday")
                if not prior.empty:
                    away_rest = max((pd.to_datetime(row["gameday"]) - pd.to_datetime(prior.iloc[-1]["gameday"])).days, 0)
            except Exception:
                pass
            rest_diff = home_rest - away_rest
            total_line = row.get("total_line", 45.0)
            if pd.isna(total_line):
                total_line = 45.0
            rows.append({
                "epa_edge": epa_edge, "spread": spread, "rest_diff": rest_diff,
                "home_off": home_off, "home_def": home_def,
                "away_off": away_off, "away_def": away_def,
                "abs_spread": abs(spread), "total_line": total_line,
                "home_covered": covered, "margin": result
            })
        df = pd.DataFrame(rows)
        if len(df) < 80:
            return None, None
        return df, epa
    except Exception:
        return None, None

@st.cache_resource(ttl=3600 * 12)
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
    spread, total_line,
    n_sims=8000,
    noise_std=11.5,
    total_adj=0.0,
    noise_extra=0.0,
    under_bias=0.0
):
    expected_margin = (home_off - away_def - (away_off - home_def)) * 35.0 + 1.2
    sim_margins = np.random.normal(loc=expected_margin, scale=noise_std + noise_extra, size=n_sims)

    expected_total = 44.0 + (home_off + away_off - home_def - away_def) * 22.0 + total_adj
    sim_totals = np.random.normal(loc=expected_total, scale=13.5 + noise_extra * 0.8, size=n_sims)

    home_cover_prob = np.mean(sim_margins > spread)
    away_cover_prob = 1.0 - home_cover_prob
    over_prob = np.mean(sim_totals > total_line) if total_line else 0.5
    # apply slight under bias from weather
    over_prob = max(0.05, min(0.95, over_prob - under_bias))
    under_prob = 1.0 - over_prob

    home_ev = home_cover_prob * 100 / 110 - (1 - home_cover_prob)
    away_ev = away_cover_prob * 100 / 110 - (1 - away_cover_prob)

    return {
        "home_cover_prob": float(home_cover_prob),
        "away_cover_prob": float(away_cover_prob),
        "over_prob": float(over_prob),
        "under_prob": float(under_prob),
        "home_ev": float(home_ev),
        "away_ev": float(away_ev),
        "expected_margin": float(expected_margin),
        "sim_margins_mean": float(np.mean(sim_margins)),
        "sim_totals_mean": float(np.mean(sim_totals))
    }

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Opportunities (ML + MC + Weather)",
    "📅 Games & Odds",
    "🎯 Player Props",
    "📊 Backtest",
    "🚀 Deploy"
])

# ========== TAB 1 ==========
with tab1:
    st.subheader("Ranked Game Opportunities – EPA + Weather + ML + Monte Carlo")
    st.caption("Score now includes dome vs outdoor + expected kickoff weather (temp / wind / precip).")

    with st.spinner("Loading data, weather forecasts & training model..."):
        team_epa = get_team_epa()
        schedules = load_schedules()
        odds_data, odds_status = fetch_nfl_odds(api_key) if api_key else (None, "No API key")
        current_season = nfl.get_current_season()
        train_seasons = list(range(current_season - 4, current_season))
        model_bundle = train_ats_model(train_seasons)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("EPA teams loaded", len(team_epa) if not team_epa.empty else 0)
    with col2:
        st.metric("Odds events", len(odds_data) if odds_data else 0)
    with col3:
        st.metric("Model trained", "Yes" if model_bundle else "No")
    st.caption(f"Odds API status: {odds_status}")

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

            if home is None or away is None:
                skipped.append(f"Unmapped: {away_full} @ {home_full}")
                continue
            if home not in team_epa.index or away not in team_epa.index:
                skipped.append(f"No EPA: {away} @ {home}")
                continue

            commence_raw = game.get("commence_time", "")
            commence = commence_raw[:16].replace("T", " ") if commence_raw else ""
            game_date = commence[:10] if commence else datetime.now().strftime("%Y-%m-%d")

            # Roof
            roof = get_roof_for_game(schedules, home, game_date)

            # Weather (only meaningful for outdoor/open)
            weather = {"temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 5.0, "source": "dome"}
            if roof not in ("dome", "closed") and home in STADIUM_COORDS:
                lat, lon = STADIUM_COORDS[home]
                weather = fetch_weather(lat, lon, commence_raw or f"{game_date}T17:00:00Z")

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

            # Rule signals
            signals = []
            rule_score = 0.0
            if epa_edge > 0.08:
                signals.append(f"Home EPA edge (+{epa_edge:.3f})")
                rule_score += 2.2
            elif epa_edge < -0.08:
                signals.append(f"Away EPA edge ({epa_edge:.3f})")
                rule_score += 2.0
            if avg_spread is not None and avg_spread > 1.5:
                signals.append("Home underdog")
                rule_score += 1.3
            if avg_spread is not None and abs(avg_spread) >= 7:
                signals.append(f"Large spread ({avg_spread:+.1f})")
                rule_score += 0.7
            if avg_total is not None and avg_total >= 48.5:
                signals.append(f"High total ({avg_total:.1f})")
                rule_score += 0.6
            if rest_diff >= 3:
                signals.append(f"Home rest +{rest_diff}d")
                rule_score += 1.1
            elif rest_diff <= -3:
                signals.append(f"Away rest {rest_diff}d")
                rule_score += 1.0

            # Weather rule points
            if wx_adj["rule_pts"] > 0:
                signals.append(wx_adj["label"])
                rule_score += wx_adj["rule_pts"]

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

            # Monte Carlo with weather adjustments
            mc = monte_carlo_game(
                home_off, home_def, away_off, away_def,
                avg_spread if avg_spread is not None else 0.0,
                avg_total,
                n_sims=n_simulations,
                total_adj=wx_adj["total_adj"],
                noise_extra=wx_adj["noise_extra"],
                under_bias=wx_adj["under_bias"]
            )

            ml_edge = abs(ml_home_cover - 0.5) * 4.0
            mc_edge = max(mc["home_ev"], mc["away_ev"]) * 8.0
            preferred_side = "Home" if (ml_home_cover > 0.5 and mc["home_cover_prob"] > 0.52) or \
                                      (ml_home_cover < 0.5 and mc["home_cover_prob"] < 0.48) else "Split"
            agreement_bonus = 1.5 if preferred_side != "Split" else 0.0
            total_score = rule_score + ml_edge + mc_edge + agreement_bonus

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

            # Weather display string
            if roof in ("dome", "closed"):
                wx_str = "Dome"
            else:
                wx_str = f"{weather['temp_f']:.0f}°F / {weather['wind_mph']:.0f} mph / {weather['precip_prob']:.0f}% precip"

            if signals or total_score > 2.0:
                opportunities.append({
                    "Game": f"{away_full} @ {home_full}",
                    "Kickoff": commence,
                    "Roof": roof.title() if roof else "—",
                    "Weather": wx_str,
                    "Spread": f"{avg_spread:+.1f}" if avg_spread is not None else "—",
                    "Total": f"{avg_total:.1f}" if avg_total is not None else "—",
                    "EPA Edge": f"{epa_edge:+.3f}",
                    "ML Home Cover %": f"{ml_home_cover*100:.1f}%",
                    "MC Home Cover %": f"{mc['home_cover_prob']*100:.1f}%",
                    "MC Over %": f"{mc['over_prob']*100:.1f}%",
                    "MC Home EV": f"{mc['home_ev']:+.3f}",
                    "Recommendation": rec,
                    "Signals": " • ".join(signals) if signals else "—",
                    "Score": round(total_score, 2)
                })

        if opportunities:
            df = pd.DataFrame(opportunities).sort_values("Score", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("#### How weather is used")
            st.markdown("""
            - **Dome / Closed roof** → neutral environment (no wind/precip effect).  
            - **Outdoor / Open** → Open-Meteo forecast at kickoff (temp, wind, precip %).  
            - High wind / high precip / extreme temp → lower expected total, extra variance, slight under bias, and rule points.  
            - These adjustments feed both the rule score and the Monte Carlo simulations.
            """)
        else:
            st.warning("No opportunities passed the score filter.")
            if skipped:
                with st.expander("Skipped games"):
                    st.write(skipped)
    else:
        if not api_key:
            st.info("Enter your The Odds API key in the sidebar.")
        elif not odds_data:
            st.error(f"Could not load odds. Status: {odds_status}")
        elif team_epa.empty:
            st.error("Could not load team EPA data from nflreadpy.")
        else:
            st.info("Waiting for data / stronger signals.")

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
        st.info("Enter API key to load games." if not api_key else f"Odds status: {odds_status}")

# ========== TAB 3 ==========
with tab3:
    st.subheader("Player Props Scanner")
    if not api_key:
        st.warning("Enter your Odds API key first.")
    elif not odds_data:
        st.info("No games available.")
    else:
        game_options = {f"{g['away_team']} @ {g['home_team']}": g["id"] for g in odds_data}
        selected_game = st.selectbox("Choose a game", options=list(game_options.keys()))
        if st.button("Load Player Props", type="primary"):
            event_id = game_options[selected_game]
            with st.spinner("Fetching..."):
                props_data = fetch_player_props(api_key, event_id)
            if props_data is None:
                st.error("Failed to fetch props.")
            elif "error" in props_data:
                st.error(f"API error: {props_data.get('error')}")
                st.write("Most free plans do not include full player props.")
            else:
                rows = []
                for book in props_data.get("bookmakers", []):
                    book_name = book.get("title", book.get("key"))
                    for market in book.get("markets", []):
                        for outcome in market.get("outcomes", []):
                            rows.append({
                                "Book": book_name,
                                "Market": market.get("key", "").replace("player_", "").replace("_", " ").title(),
                                "Player": outcome.get("description", outcome.get("name", "")),
                                "Side": outcome.get("name"),
                                "Line": outcome.get("point"),
                                "Odds": outcome.get("price")
                            })
                if rows:
                    st.success(f"Found {len(rows)} prop lines")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.warning("No player props returned.")

# ========== TAB 4 ==========
with tab4:
    st.subheader("Backtest – EPA + ML")
    min_edge = st.slider("Minimum EPA edge", 0.03, 0.20, 0.05, 0.01)
    seasons_back = st.multiselect("Eval seasons", [2021, 2022, 2023, 2024, 2025], default=[2023, 2024, 2025])
    train_on = st.multiselect("Train seasons", [2019, 2020, 2021, 2022, 2023, 2024], default=[2020, 2021, 2022])
    if st.button("Run Backtest"):
        with st.spinner("Running..."):
            try:
                model_bundle = train_ats_model(train_on)
                if model_bundle is None:
                    st.error("Not enough data to train.")
                else:
                    model, feature_cols = model_bundle
                    hist_sched = load_schedules(seasons=seasons_back)
                    hist_epa = get_team_epa(seasons=seasons_back)
                    completed = hist_sched[hist_sched["result"].notna() & hist_sched["spread_line"].notna()]
                    results = []
                    for _, row in completed.iterrows():
                        home, away = row["home_team"], row["away_team"]
                        if home not in hist_epa.index or away not in hist_epa.index:
                            continue
                        home_off = hist_epa.loc[home, "off_epa"]
                        home_def = hist_epa.loc[home, "def_epa"]
                        away_off = hist_epa.loc[away, "off_epa"]
                        away_def = hist_epa.loc[away, "def_epa"]
                        epa_edge = (home_off - away_def) - (away_off - home_def)
                        spread = float(row["spread_line"])
                        result = float(row["result"])
                        total_line = row.get("total_line", 45.0)
                        if pd.isna(total_line):
                            total_line = 45.0
                        feat = pd.DataFrame([{
                            "epa_edge": epa_edge, "spread": spread, "rest_diff": 0,
                            "home_off": home_off, "home_def": home_def,
                            "away_off": away_off, "away_def": away_def,
                            "abs_spread": abs(spread), "total_line": total_line
                        }])[feature_cols]
                        ml_prob = float(model.predict_proba(feat)[0, 1])
                        side = None
                        if epa_edge >= min_edge and ml_prob > 0.52:
                            side = "Home"
                            covered = result > spread
                        elif epa_edge <= -min_edge and ml_prob < 0.48:
                            side = "Away"
                            covered = result < spread
                        if side:
                            results.append({"side": side, "covered": covered, "ml_prob": ml_prob, "epa_edge": epa_edge})
                    if results:
                        res_df = pd.DataFrame(results)
                        st.metric("ATS Win Rate", f"{res_df['covered'].mean():.1%}", delta=f"{len(res_df)} bets")
                    else:
                        st.warning("No qualifying games.")
            except Exception as e:
                st.error(str(e))

with tab5:
    st.write("Deployment notes can go here.")
