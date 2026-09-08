import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import nflreadpy as nfl
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NFL Opportunity Scanner – Quant Edition",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 NFL Opportunity Scanner – Quant Edition")
st.caption("10-Year Recency-Weighted Power Ratings • ML Win Probability • Monte Carlo Simulation")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Settings")
api_key = st.sidebar.text_input(
    "The Odds API Key",
    type="password",
    help="Get a free key at https://the-odds-api.com"
)
st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
LOOKBACK_YEARS = st.sidebar.slider("Years of history to train on", 3, 10, 10)
SEASON_HALF_LIFE = st.sidebar.slider("Season recency half-life (years)", 0.5, 5.0, 2.0, 0.5,
                                      help="Lower = more weight on recent seasons")
N_SIMS = st.sidebar.select_slider("Monte Carlo simulations per game", options=[2000, 5000, 10000, 20000, 50000], value=20000)
CLASSIFIER_TYPE = st.sidebar.selectbox("Win probability model", ["Logistic Regression", "Gradient Boosting"])
MARGIN_MODEL_TYPE = st.sidebar.selectbox("Margin/Total model", ["Ridge Regression", "Gradient Boosting"])
st.sidebar.markdown("---")
st.sidebar.info("Player Props usually need a higher plan on The Odds API.")
st.sidebar.caption(
    "⚠️ This tool is for research/entertainment purposes. No model guarantees profit. "
    "Bet responsibly."
)

# =========================================================
# CONSTANTS / HELPERS
# =========================================================
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
# Handle franchise relocations so historical games map to current abbreviation
TEAM_ALIAS = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

def to_abbr(name):
    abbr = TEAM_NAME_MAP.get(name, name)
    return TEAM_ALIAS.get(abbr, abbr)

def normalize_team(abbr):
    return TEAM_ALIAS.get(abbr, abbr)

def get_current_nfl_season():
    today = datetime.now()
    if today.month < 3:
        return today.year - 1
    return today.year

def american_to_prob(odds):
    """Convert American odds to implied probability."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)

def devig_two_way(prob_a, prob_b):
    """Remove vig from a two-way market by normalizing implied probabilities."""
    if prob_a is None or prob_b is None:
        return prob_a, prob_b
    total = prob_a + prob_b
    if total <= 0:
        return prob_a, prob_b
    return prob_a / total, prob_b / total

def season_weight(season, current_season, half_life):
    """Exponential recency weight by season."""
    years_back = max(current_season - season, 0)
    return 0.5 ** (years_back / half_life)

# =========================================================
# DATA LOADERS
# =========================================================
@st.cache_data(ttl=3600 * 6)
def load_schedules_range(years_back, current_season=None):
    if current_season is None:
        current_season = get_current_nfl_season()
    seasons = list(range(current_season - years_back + 1, current_season + 1))
    try:
        sched = nfl.load_schedules(seasons=seasons)
        sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
        sched["home_team"] = sched["home_team"].apply(normalize_team)
        sched["away_team"] = sched["away_team"].apply(normalize_team)
        return sched
    except Exception as e:
        st.warning(f"Schedule load error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600 * 6)
def load_pbp_range(years_back, current_season=None):
    if current_season is None:
        current_season = get_current_nfl_season()
    seasons = list(range(current_season - years_back + 1, current_season + 1))
    try:
        pbp = nfl.load_pbp(seasons=seasons)
        pbp = pbp.to_pandas() if hasattr(pbp, "to_pandas") else pbp
        pbp = pbp[
            (pbp["play_type"].isin(["pass", "run"])) &
            (pbp["epa"].notna()) &
            (pbp["posteam"].notna()) &
            (pbp["defteam"].notna())
        ].copy()
        pbp["posteam"] = pbp["posteam"].apply(normalize_team)
        pbp["defteam"] = pbp["defteam"].apply(normalize_team)
        return pbp
    except Exception as e:
        st.warning(f"PBP load error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600 * 4)
def load_recent_player_stats(years_back=3):
    current = get_current_nfl_season()
    seasons = list(range(current - years_back + 1, current + 1))
    try:
        stats = nfl.load_player_stats(seasons=seasons, summary_level="week")
        return stats.to_pandas() if hasattr(stats, "to_pandas") else stats
    except Exception:
        return pd.DataFrame()

def fetch_nfl_odds(key):
    if not key:
        return None
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {"apiKey": key, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"}
    try:
        r = requests.get(url, params=params, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def fetch_player_props(key, event_id):
    if not key or not event_id:
        return None
    markets = ",".join([
        "player_pass_yds", "player_pass_tds", "player_rush_yds",
        "player_reception_yds", "player_receptions", "player_anytime_td",
        "player_pass_completions", "player_rush_tds", "player_reception_tds"
    ])
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {"apiKey": key, "regions": "us", "markets": markets, "oddsFormat": "american"}
    try:
        r = requests.get(url, params=params, timeout=25)
        return r.json() if r.status_code == 200 else {"error": f"Status {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# =========================================================
# EPA METRICS (season-weighted + recent form)
# =========================================================
@st.cache_data(ttl=3600 * 6)
def build_epa_metrics(pbp, current_season, half_life):
    if pbp.empty:
        return pd.DataFrame()

    pbp = pbp.copy()
    pbp["season_w"] = pbp["season"].apply(lambda s: season_weight(s, current_season, half_life))

    def wavg(df, group_col, weight_col="season_w", value_col="epa"):
        def _f(g):
            w = g[weight_col]
            return np.average(g[value_col], weights=w) if w.sum() > 0 else g[value_col].mean()
        return df.groupby(group_col).apply(_f)

    pass_pbp = pbp[pbp["play_type"] == "pass"]
    rush_pbp = pbp[pbp["play_type"] == "run"]

    off_pass = wavg(pass_pbp, "posteam").rename("off_pass_epa")
    off_rush = wavg(rush_pbp, "posteam").rename("off_rush_epa")
    def_pass = wavg(pass_pbp, "defteam").rename("def_pass_epa")
    def_rush = wavg(rush_pbp, "defteam").rename("def_rush_epa")
    season_level = pd.concat([off_pass, off_rush, def_pass, def_rush], axis=1)

    # Recent form: only current season, last 6 weeks, weighted toward most recent
    cur = pbp[pbp["season"] == current_season]
    if "week" in cur.columns and not cur.empty:
        max_week = cur["week"].max()
        recent = cur[cur["week"] >= max(1, max_week - 5)].copy()
        recent["form_w"] = recent["week"].apply(lambda w: 0.5 ** ((max_week - w) / 3))
    else:
        recent = cur.tail(int(len(cur) * 0.25)).copy()
        recent["form_w"] = 1.0

    if not recent.empty:
        rp = recent[recent["play_type"] == "pass"]
        rr = recent[recent["play_type"] == "run"]
        off_pass_r = wavg(rp, "posteam", "form_w").rename("off_pass_recent")
        off_rush_r = wavg(rr, "posteam", "form_w").rename("off_rush_recent")
        def_pass_r = wavg(rp, "defteam", "form_w").rename("def_pass_recent")
        def_rush_r = wavg(rr, "defteam", "form_w").rename("def_rush_recent")
        recent_level = pd.concat([off_pass_r, off_rush_r, def_pass_r, def_rush_r], axis=1)
    else:
        recent_level = pd.DataFrame()

    return season_level.join(recent_level, how="outer").fillna(0)

# =========================================================
# POWER RATINGS — regularized (ridge) Massey-style regression
# Weighted by season recency, solved on point margin.
# =========================================================
@st.cache_data(ttl=3600 * 6)
def build_power_ratings(schedules, current_season, half_life, alpha=45.0):
    games = schedules[schedules["home_score"].notna() & schedules["away_score"].notna()].copy()
    if games.empty:
        return {}, 0.0, pd.DataFrame()

    games["margin"] = games["home_score"] - games["away_score"]
    games["weight"] = games["season"].apply(lambda s: season_weight(s, current_season, half_life))
    # extra decay within a season so week 1 counts a little less than week 18 of the same year
    if "week" in games.columns:
        games["weight"] *= games.groupby("season")["week"].transform(
            lambda w: 0.85 + 0.15 * (w - w.min()) / max((w.max() - w.min()), 1)
        )

    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(games)
    X = np.zeros((n, len(teams) + 1))
    y = games["margin"].values
    w = games["weight"].values

    for i, (_, row) in enumerate(games.iterrows()):
        X[i, idx[row["home_team"]]] = 1.0
        X[i, idx[row["away_team"]]] = -1.0
        X[i, -1] = 1.0  # home-field column

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, y, sample_weight=w)

    ratings = {t: model.coef_[idx[t]] for t in teams}
    home_field_adv = model.coef_[-1]

    preds = model.predict(X)
    resid_std = float(np.sqrt(np.average((y - preds) ** 2, weights=w)))

    ratings_df = pd.DataFrame({"team": teams, "power_rating": [ratings[t] for t in teams]})
    ratings_df = ratings_df.sort_values("power_rating", ascending=False).reset_index(drop=True)
    ratings_df["rank"] = ratings_df.index + 1

    return ratings, home_field_adv, ratings_df, resid_std

# =========================================================
# REST DAYS
# =========================================================
def get_rest_days(schedules, team, game_date):
    try:
        mask = ((schedules["home_team"] == team) | (schedules["away_team"] == team)) & \
               (schedules["gameday"] < str(game_date))
        team_games = schedules[mask].sort_values("gameday")
        if team_games.empty:
            return 7
        last = team_games.iloc[-1]["gameday"]
        return max((pd.to_datetime(game_date) - pd.to_datetime(last)).days, 0)
    except Exception:
        return 7

# =========================================================
# TRAIN ML MODELS (win probability + margin + total)
# Trained on historical games, features built from power
# ratings + EPA at time-of-season (approximated with season-level
# EPA joined per game — a simplification vs a true walk-forward
# feature set, called out in the Backtest tab).
# =========================================================
@st.cache_data(ttl=3600 * 6)
def build_training_table(schedules, epa_metrics, ratings, home_field_adv, current_season, half_life):
    games = schedules[schedules["home_score"].notna() & schedules["away_score"].notna()].copy()
    if games.empty or epa_metrics.empty:
        return pd.DataFrame()

    rows = []
    for _, row in games.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in ratings or a not in ratings:
            continue
        if h not in epa_metrics.index or a not in epa_metrics.index:
            continue
        m = epa_metrics
        pass_edge = (m.loc[h, "off_pass_epa"] - m.loc[a, "def_pass_epa"]) - \
                    (m.loc[a, "off_pass_epa"] - m.loc[h, "def_pass_epa"])
        rush_edge = (m.loc[h, "off_rush_epa"] - m.loc[a, "def_rush_epa"]) - \
                    (m.loc[a, "off_rush_epa"] - m.loc[h, "def_rush_epa"])
        rating_diff = ratings[h] - ratings[a]
        margin = row["home_score"] - row["away_score"]
        total_pts = row["home_score"] + row["away_score"]
        rows.append({
            "season": row["season"],
            "rating_diff": rating_diff,
            "pass_edge": pass_edge,
            "rush_edge": rush_edge,
            "margin": margin,
            "home_win": int(margin > 0),
            "total_pts": total_pts,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["weight"] = df["season"].apply(lambda s: season_weight(s, current_season, half_life))
    return df

@st.cache_resource(ttl=3600 * 6)
def train_models(_train_df, classifier_type, margin_model_type):
    """Underscore prefix on _train_df tells st.cache_resource not to hash the (large) dataframe by value issues;
    we still pass it in normally since cache_resource hashes objects by id for mutable types is avoided by Streamlit
    for DataFrames automatically in most versions — kept simple here."""
    train_df = _train_df
    if train_df is None or train_df.empty or len(train_df) < 50:
        return None

    features = ["rating_diff", "pass_edge", "rush_edge"]
    X = train_df[features].values
    w = train_df["weight"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    y_win = train_df["home_win"].values
    y_margin = train_df["margin"].values
    y_total = train_df["total_pts"].values

    if classifier_type == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05)
    else:
        clf = LogisticRegression(max_iter=1000)
    clf.fit(Xs, y_win, sample_weight=w)

    if margin_model_type == "Gradient Boosting":
        reg_margin = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05)
        reg_total = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05)
    else:
        reg_margin = Ridge(alpha=1.0)
        reg_total = Ridge(alpha=1.0)
    reg_margin.fit(Xs, y_margin, sample_weight=w)
    reg_total.fit(Xs, y_total, sample_weight=w)

    margin_resid_std = float(np.sqrt(np.average((y_margin - reg_margin.predict(Xs)) ** 2, weights=w)))
    total_resid_std = float(np.sqrt(np.average((y_total - reg_total.predict(Xs)) ** 2, weights=w)))

    return {
        "scaler": scaler,
        "clf": clf,
        "reg_margin": reg_margin,
        "reg_total": reg_total,
        "margin_resid_std": margin_resid_std,
        "total_resid_std": total_resid_std,
        "features": features,
    }

# =========================================================
# MONTE CARLO SIMULATION
# =========================================================
def monte_carlo_game(models, feature_row, n_sims, home_spread=None, total_line=None):
    """
    Simulates a single game n_sims times using the regression model's
    predicted margin/total plus their residual distributions (captures
    the uncertainty the regression itself doesn't explain), rather than
    just taking a point estimate.
    """
    X = np.array([feature_row])
    Xs = models["scaler"].transform(X)

    pred_margin = models["reg_margin"].predict(Xs)[0]
    pred_total = models["reg_total"].predict(Xs)[0]
    margin_sigma = models["margin_resid_std"]
    total_sigma = models["total_resid_std"]

    # Margin and total are simulated with a modest positive correlation:
    # blowouts tend to slightly inflate total scoring. rho is intentionally
    # conservative since this isn't estimated from data here.
    rho = 0.15
    z1 = np.random.normal(0, 1, n_sims)
    z2 = np.random.normal(0, 1, n_sims)
    z2_corr = rho * z1 + np.sqrt(1 - rho ** 2) * z2

    margin_sims = pred_margin + margin_sigma * z1
    total_sims = np.clip(pred_total + total_sigma * z2_corr, 10, None)

    home_score_sims = (total_sims + margin_sims) / 2
    away_score_sims = (total_sims - margin_sims) / 2

    result = {
        "pred_margin": pred_margin,
        "pred_total": pred_total,
        "home_win_prob": float(np.mean(margin_sims > 0)),
        "margin_p10": float(np.percentile(margin_sims, 10)),
        "margin_p50": float(np.percentile(margin_sims, 50)),
        "margin_p90": float(np.percentile(margin_sims, 90)),
        "total_p10": float(np.percentile(total_sims, 10)),
        "total_p50": float(np.percentile(total_sims, 50)),
        "total_p90": float(np.percentile(total_sims, 90)),
        "avg_home_score": float(np.mean(home_score_sims)),
        "avg_away_score": float(np.mean(away_score_sims)),
    }

    if home_spread is not None:
        # home_spread follows market convention: negative = home favored.
        # Home "covers" if margin_sim > -home_spread.
        result["home_cover_prob"] = float(np.mean(margin_sims > -home_spread))
    if total_line is not None:
        result["over_prob"] = float(np.mean(total_sims > total_line))
        result["under_prob"] = float(np.mean(total_sims < total_line))

    return result

# =========================================================
# PLAYER PROP HELPERS
# =========================================================
MARKET_TO_STAT = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
    "player_pass_completions": "completions",
    "player_rush_tds": "rushing_tds",
    "player_reception_tds": "receiving_tds",
}

def get_player_recent_stats(player_stats, player_name, stat_col, last_n=10):
    """Recency-weighted mean + std of a player's recent games (half-life ~4 games)."""
    if player_stats.empty or not player_name or stat_col not in player_stats.columns:
        return None, None, 0
    mask = (
        player_stats["player_display_name"].str.contains(player_name, case=False, na=False) |
        player_stats["player_name"].str.contains(player_name.split()[-1], case=False, na=False)
    )
    pdf = player_stats[mask].copy()
    if pdf.empty:
        return None, None, 0
    if "week" in pdf.columns and "season" in pdf.columns:
        pdf = pdf.sort_values(["season", "week"], ascending=False)
    recent = pdf.head(last_n).reset_index(drop=True)
    values = recent[stat_col].dropna()
    if len(values) == 0:
        return None, None, 0
    weights = 0.5 ** (np.arange(len(values)) / 4.0)  # game 0 (most recent) weighted highest
    weighted_mean = float(np.average(values, weights=weights[:len(values)]))
    std = float(values.std()) if len(values) > 1 else weighted_mean * 0.3
    return weighted_mean, std, len(values)

def prop_monte_carlo(mean, std, line, side, n_sims=20000):
    if std is None or std <= 0:
        std = max(mean * 0.3, 1.0)
    sims = np.random.normal(mean, std, n_sims)
    sims = np.clip(sims, 0, None)
    over_prob = float(np.mean(sims > line))
    return over_prob if side == "Over" else 1 - over_prob

WEEK_OPTIONS = [f"Week {i}" for i in range(1, 19)] + [
    "Wild Card", "Divisional", "Conference Championship", "Super Bowl"
]

def filter_schedule_by_week(sched, selected, season):
    if sched.empty:
        return pd.DataFrame()
    s = sched[sched["season"] == season]
    if selected.startswith("Week"):
        return s[s["week"] == int(selected.split()[1])].copy()
    tag = {"Wild Card": "WC", "Divisional": "DIV", "Conference Championship": "CON", "Super Bowl": "SB"}.get(selected)
    return s[s["game_type"] == tag].copy() if tag else pd.DataFrame()

# =========================================================
# LOAD SHARED DATA (used across tabs)
# =========================================================
current_season = get_current_nfl_season()

with st.spinner(f"Loading {LOOKBACK_YEARS} seasons of data and training models..."):
    schedules_all = load_schedules_range(LOOKBACK_YEARS, current_season)
    pbp_all = load_pbp_range(min(LOOKBACK_YEARS, 6), current_season)  # PBP is heavy; cap EPA lookback
    epa_metrics = build_epa_metrics(pbp_all, current_season, SEASON_HALF_LIFE)

    ratings, home_field_adv, ratings_df, power_resid_std = ({}, 0.0, pd.DataFrame(), 0.0)
    if not schedules_all.empty:
        ratings, home_field_adv, ratings_df, power_resid_std = build_power_ratings(
            schedules_all, current_season, SEASON_HALF_LIFE
        )

    train_df = build_training_table(schedules_all, epa_metrics, ratings, home_field_adv,
                                     current_season, SEASON_HALF_LIFE)
    models = train_models(train_df, CLASSIFIER_TYPE, MARGIN_MODEL_TYPE) if not train_df.empty else None

odds_data = fetch_nfl_odds(api_key) if api_key else None

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Opportunities", "📈 Power Ratings", "📅 Games & Odds",
    "🎯 Player Props", "📊 Backtest", "🚀 Deploy"
])

# ---------- TAB 1: OPPORTUNITIES ----------
with tab1:
    st.subheader("Model vs Market — Monte Carlo Edges")
    st.caption(f"Trained on {LOOKBACK_YEARS} seasons, season half-life = {SEASON_HALF_LIFE}y, {N_SIMS:,} sims/game")

    if models is None:
        st.error("Not enough historical data to train models yet. Try increasing lookback years.")
    else:
        selected_week = st.selectbox("Select Week / Round", options=WEEK_OPTIONS, index=0)
        week_games = filter_schedule_by_week(schedules_all, selected_week, current_season)

        if week_games.empty:
            st.info(f"No games found for {selected_week} in the {current_season} season yet.")
        else:
            odds_lookup = {}
            if odds_data:
                for g in odds_data:
                    odds_lookup[(g.get("away_team"), g.get("home_team"))] = g

            rows = []
            for _, row in week_games.iterrows():
                home_full, away_full = row["home_team"], row["away_team"]
                home, away = to_abbr(home_full), to_abbr(away_full)
                gameday = str(row.get("gameday", ""))[:10]

                if home not in ratings or away not in ratings:
                    continue
                if home not in epa_metrics.index or away not in epa_metrics.index:
                    continue

                m = epa_metrics
                pass_edge = (m.loc[home, "off_pass_epa"] - m.loc[away, "def_pass_epa"]) - \
                            (m.loc[away, "off_pass_epa"] - m.loc[home, "def_pass_epa"])
                rush_edge = (m.loc[home, "off_rush_epa"] - m.loc[away, "def_rush_epa"]) - \
                            (m.loc[away, "off_rush_epa"] - m.loc[home, "def_rush_epa"])
                rating_diff = ratings[home] - ratings[away]

                feature_row = [rating_diff, pass_edge, rush_edge]

                game_odds = odds_lookup.get((away_full, home_full))
                market_spread, market_total, home_ml, away_ml = None, None, None, None
                if game_odds:
                    for book in game_odds.get("bookmakers", []):
                        for mk in book.get("markets", []):
                            if mk["key"] == "spreads":
                                for o in mk["outcomes"]:
                                    if o["name"] == home_full and market_spread is None:
                                        market_spread = o.get("point")
                            if mk["key"] == "totals":
                                for o in mk["outcomes"]:
                                    if o["name"] == "Over" and market_total is None:
                                        market_total = o.get("point")
                            if mk["key"] == "h2h":
                                for o in mk["outcomes"]:
                                    if o["name"] == home_full and home_ml is None:
                                        home_ml = o.get("price")
                                    if o["name"] == away_full and away_ml is None:
                                        away_ml = o.get("price")

                sim = monte_carlo_game(models, feature_row, N_SIMS, market_spread, market_total)

                market_home_prob, market_away_prob = None, None
                if home_ml is not None and away_ml is not None:
                    market_home_prob, market_away_prob = devig_two_way(
                        american_to_prob(home_ml), american_to_prob(away_ml)
                    )

                edge_vs_market = None
                if market_home_prob is not None:
                    edge_vs_market = sim["home_win_prob"] - market_home_prob

                rows.append({
                    "Game": f"{away_full} @ {home_full}",
                    "Date": gameday,
                    "Model Win% (Home)": f"{sim['home_win_prob']*100:.1f}%",
                    "Market Win% (Home)": f"{market_home_prob*100:.1f}%" if market_home_prob else "—",
                    "ML Edge": f"{edge_vs_market*100:+.1f}%" if edge_vs_market is not None else "—",
                    "Model Margin (Home)": f"{sim['pred_margin']:+.1f}",
                    "Market Spread (Home)": f"{market_spread:+.1f}" if market_spread is not None else "—",
                    "Cover% (Home)": f"{sim.get('home_cover_prob', 0)*100:.1f}%" if market_spread is not None else "—",
                    "Model Total": f"{sim['pred_total']:.1f}",
                    "Market Total": f"{market_total:.1f}" if market_total is not None else "—",
                    "Over%": f"{sim.get('over_prob', 0)*100:.1f}%" if market_total is not None else "—",
                    "_edge_sort": abs(edge_vs_market) if edge_vs_market is not None else abs(sim["home_win_prob"] - 0.5),
                })

            if rows:
                df = pd.DataFrame(rows).sort_values("_edge_sort", ascending=False).drop(columns="_edge_sort")
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.success(
                    "ML Edge = model win probability minus de-vigged market win probability. "
                    "Cover%/Over% come from Monte Carlo simulation against the current market line."
                )
                if not api_key:
                    st.info("Add an Odds API key in the sidebar to see market lines and edges vs. the market.")
            else:
                st.info("No games could be matched to ratings/EPA data for this week.")

# ---------- TAB 2: POWER RATINGS ----------
with tab2:
    st.subheader("Power Ratings (Ridge-Regularized Massey Ratings)")
    st.caption(
        f"Estimated from {LOOKBACK_YEARS} seasons of results, weighted so recent seasons "
        f"count more (half-life = {SEASON_HALF_LIFE} years). Value = expected point margin vs. an average team on a neutral field."
    )
    if ratings_df.empty:
        st.info("No ratings available yet.")
    else:
        display_df = ratings_df.copy()
        display_df["power_rating"] = display_df["power_rating"].round(2)
        st.dataframe(display_df[["rank", "team", "power_rating"]], use_container_width=True, hide_index=True)
        st.caption(f"Estimated home-field advantage: {home_field_adv:+.2f} points • Residual std: {power_resid_std:.1f} pts")

# ---------- TAB 3: GAMES & ODDS ----------
with tab3:
    st.subheader("Upcoming Games & Current Lines")
    if odds_data:
        rows = []
        for g in odds_data:
            home, away = g["home_team"], g["away_team"]
            commence = g.get("commence_time", "")[:16].replace("T", " ")
            spread = total = "—"
            for book in g.get("bookmakers", [])[:1]:
                for mk in book.get("markets", []):
                    if mk["key"] == "spreads":
                        for o in mk["outcomes"]:
                            if o["name"] == home:
                                spread = f"{o.get('point', 0):+.1f}"
                    if mk["key"] == "totals":
                        for o in mk["outcomes"]:
                            if o["name"] == "Over":
                                total = f"{o.get('point', 0):.1f}"
            rows.append({"Away": away, "Home": home, "Kickoff": commence, "Spread": spread, "Total": total})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Enter an Odds API key in the sidebar to load live lines.")

# ---------- TAB 4: PLAYER PROPS ----------
with tab4:
    st.subheader("Player Props Opportunity Scanner (Monte Carlo)")
    st.write("Select a game to load available player props and simulate them against recency-weighted recent performance.")

    if not api_key:
        st.warning("Enter your Odds API key in the sidebar first.")
    elif not odds_data:
        st.info("No games available right now.")
    else:
        game_options = {f"{g['away_team']} @ {g['home_team']}": g["id"] for g in odds_data}
        selected_game = st.selectbox("Choose a game", options=list(game_options.keys()))

        if st.button("Scan Player Props Opportunities", type="primary"):
            event_id = game_options[selected_game]
            with st.spinner("Fetching props + running simulations..."):
                props_data = fetch_player_props(api_key, event_id)
                player_stats = load_recent_player_stats()

            if props_data is None:
                st.error("Failed to fetch props.")
            elif "error" in props_data:
                st.error(f"API error: {props_data.get('error')}")
                st.write("Most free/basic plans do not include full NFL player props.")
            else:
                opportunities = []
                for book in props_data.get("bookmakers", []):
                    book_name = book.get("title", book.get("key"))
                    for market in book.get("markets", []):
                        market_key = market.get("key", "")
                        if market_key not in MARKET_TO_STAT:
                            continue
                        stat_col = MARKET_TO_STAT[market_key]
                        market_label = market_key.replace("player_", "").replace("_", " ").title()

                        for outcome in market.get("outcomes", []):
                            player = outcome.get("description") or outcome.get("name", "")
                            side = outcome.get("name")
                            line = outcome.get("point")
                            odds = outcome.get("price")
                            if not player or line is None or side not in ("Over", "Under"):
                                continue

                            mean, std, sample = get_player_recent_stats(player_stats, player, stat_col)
                            if mean is None:
                                continue

                            model_prob = prop_monte_carlo(mean, std, line, side, n_sims=10000)
                            implied_prob = american_to_prob(odds)
                            edge = (model_prob - implied_prob) if implied_prob else None

                            score = 0.0
                            if edge is not None:
                                score = edge * 10
                            if sample >= 8:
                                score += 0.5
                            elif sample >= 4:
                                score += 0.2

                            if edge is not None and edge > 0.02:
                                opportunities.append({
                                    "Player": player,
                                    "Market": market_label,
                                    "Side": side,
                                    "Line": line,
                                    "Recency-Wtd Avg": round(mean, 1),
                                    "Model Prob": f"{model_prob*100:.1f}%",
                                    "Implied Prob": f"{implied_prob*100:.1f}%" if implied_prob else "—",
                                    "Edge": f"{edge*100:+.1f}%",
                                    "Sample": sample,
                                    "Odds": odds,
                                    "Book": book_name,
                                    "Score": round(score, 2),
                                })

                if opportunities:
                    opp_df = pd.DataFrame(opportunities).sort_values("Score", ascending=False)
                    opp_df = opp_df.drop_duplicates(subset=["Player", "Market", "Side"], keep="first")
                    st.success(f"Found {len(opp_df)} opportunities with positive modeled edge")
                    st.dataframe(opp_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No props showed a positive edge vs. the simulated distribution.")

# ---------- TAB 5: BACKTEST ----------
with tab5:
    st.subheader("Walk-Forward-Style Backtest")
    st.caption(
        "Note: this backtest reuses the full-history-trained power ratings/EPA rather than a strict "
        "walk-forward refit per week, so treat results as directional, not a guarantee of future edge."
    )
    min_edge_pct = st.slider("Minimum model-vs-50% edge to flag a bet", 1, 20, 5, 1)
    backtest_years = st.slider("Years to backtest over", 1, min(LOOKBACK_YEARS, 10), min(5, LOOKBACK_YEARS))

    if st.button("Run Backtest"):
        if models is None or train_df.empty:
            st.error("Models aren't trained yet — need more historical data.")
        else:
            with st.spinner("Running backtest..."):
                bt = train_df[train_df["season"] >= current_season - backtest_years + 1].copy()
                X = bt[models["features"]].values
                Xs = models["scaler"].transform(X)
                win_probs = models["clf"].predict_proba(Xs)[:, 1] if hasattr(models["clf"], "predict_proba") else models["clf"].predict(Xs)
                bt["model_home_win_prob"] = win_probs
                bt["edge"] = (bt["model_home_win_prob"] - 0.5).abs()
                flagged = bt[bt["edge"] >= (min_edge_pct / 100)].copy()
                flagged["predicted_home_win"] = flagged["model_home_win_prob"] > 0.5
                flagged["correct"] = flagged["predicted_home_win"] == (flagged["home_win"] == 1)

                if not flagged.empty:
                    st.metric("Accuracy on flagged games", f"{flagged['correct'].mean()*100:.1f}%",
                               delta=f"{len(flagged)} games")
                    st.caption("Accuracy on ALL games (unflagged baseline):")
                    bt["predicted_home_win"] = bt["model_home_win_prob"] > 0.5
                    bt["correct"] = bt["predicted_home_win"] == (bt["home_win"] == 1)
                    st.metric("Overall accuracy", f"{bt['correct'].mean()*100:.1f}%", delta=f"{len(bt)} games")
                else:
                    st.warning("No games met the minimum edge threshold.")

# ---------- TAB 6: DEPLOY ----------
with tab6:
    st.subheader("Deploy / Update")
    st.markdown("""
    After making changes:
    1. Upload the new `app.py` to your GitHub repo
    2. Make sure `requirements.txt` includes: `streamlit pandas numpy requests nflreadpy scikit-learn`
    3. Go to share.streamlit.io → Reboot the app

    **What changed in this version:**
    - Power ratings are now a ridge-regularized Massey-style regression over up to 10 seasons of
      results, with an exponential recency weight by season (adjustable half-life) instead of a flat average.
    - Win probability comes from a trained classifier (Logistic Regression or Gradient Boosting) using
      power-rating differential + weighted EPA splits as features, rather than hand-tuned point bonuses.
    - Point margin and total points are predicted by a regression model (Ridge or Gradient Boosting),
      and fed into a Monte Carlo simulation (using the model's own residual error) to get win/cover/over
      probabilities with uncertainty baked in, instead of a single point estimate.
    - Player props are scored the same way: recency-weighted recent performance → simulated distribution
      → probability compared against the sportsbook's de-vigged implied probability.
    """)

st.sidebar.markdown("---")
st.sidebar.caption(f"Quant Edition • {current_season} season • Trained on {LOOKBACK_YEARS}yr history")
