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

st.set_page_config(page_title="NFL Betting Edges", page_icon="🏈", layout="wide")

st.title("🏈 Upcoming NFL Games — Top Model Edges")
st.caption("Integrated Power Ratings • Machine Learning • Monte Carlo Simulation")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("The Odds API Key", type="password", help="Get key at https://the-odds-api.com")
LOOKBACK_YEARS = st.sidebar.slider("Years of history to train on", 3, 10, 5)
SEASON_HALF_LIFE = st.sidebar.slider("Season recency half-life (years)", 0.5, 5.0, 2.0, 0.5)
N_SIMS = st.sidebar.select_slider("Monte Carlo simulations per game", options=[5000, 10000, 20000], value=10000)

# =========================================================
# HELPERS
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
TEAM_ALIAS = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

def to_abbr(name):
    abbr = TEAM_NAME_MAP.get(name, name)
    return TEAM_ALIAS.get(abbr, abbr)

def normalize_team(abbr):
    return TEAM_ALIAS.get(abbr, abbr)

def get_current_nfl_season():
    today = datetime.now()
    return today.year - 1 if today.month < 3 else today.year

def american_to_prob(odds):
    try:
        odds = float(odds)
        return 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)
    except (TypeError, ValueError):
        return None

def devig_two_way(prob_a, prob_b):
    if prob_a is None or prob_b is None:
        return prob_a, prob_b
    total = prob_a + prob_b
    return (prob_a / total, prob_b / total) if total > 0 else (prob_a, prob_b)

def season_weight(season, current_season, half_life):
    return 0.5 ** (max(current_season - season, 0) / half_life)

# =========================================================
# DATA LOADERS & ENGINE
# =========================================================
@st.cache_data(ttl=3600 * 6)
def load_data(years_back, current_season):
    seasons = list(range(current_season - years_back + 1, current_season + 1))
    
    # Load Schedules
    sched = nfl.load_schedules(seasons=seasons)
    sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    sched["home_team"] = sched["home_team"].apply(normalize_team)
    sched["away_team"] = sched["away_team"].apply(normalize_team)

    # Load Play-by-Play for EPA
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

    return sched, pbp

def build_power_ratings(schedules, current_season, half_life, alpha=45.0):
    games = schedules[schedules["home_score"].notna() & schedules["away_score"].notna()].copy()
    if games.empty:
        return {}
    games["margin"] = games["home_score"] - games["away_score"]
    games["weight"] = games["season"].apply(lambda s: season_weight(s, current_season, half_life))

    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(games), len(teams) + 1))
    y = games["margin"].values
    w = games["weight"].values

    for i, (_, row) in enumerate(games.iterrows()):
        X[i, idx[row["home_team"]]] = 1.0
        X[i, idx[row["away_team"]]] = -1.0
        X[i, -1] = 1.0

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, y, sample_weight=w)
    return {t: model.coef_[idx[t]] for t in teams}

def build_epa_metrics(pbp, current_season, half_life):
    if pbp.empty:
        return pd.DataFrame()
    pbp["season_w"] = pbp["season"].apply(lambda s: season_weight(s, current_season, half_life))
    
    def wavg(df, group_col):
        return df.groupby(group_col).apply(
            lambda g: np.average(g["epa"], weights=g["season_w"]) if g["season_w"].sum() > 0 else g["epa"].mean()
        )

    pass_pbp = pbp[pbp["play_type"] == "pass"]
    rush_pbp = pbp[pbp["play_type"] == "run"]

    off_pass = wavg(pass_pbp, "posteam").rename("off_pass_epa")
    off_rush = wavg(rush_pbp, "posteam").rename("off_rush_epa")
    def_pass = wavg(pass_pbp, "defteam").rename("def_pass_epa")
    def_rush = wavg(rush_pbp, "defteam").rename("def_rush_epa")

    return pd.concat([off_pass, off_rush, def_pass, def_rush], axis=1).fillna(0)

def train_models(schedules, epa_metrics, ratings, current_season, half_life):
    games = schedules[schedules["home_score"].notna() & schedules["away_score"].notna()].copy()
    rows = []
    for _, row in games.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h in ratings and a in ratings and h in epa_metrics.index and a in epa_metrics.index:
            pass_edge = (epa_metrics.loc[h, "off_pass_epa"] - epa_metrics.loc[a, "def_pass_epa"]) - \
                        (epa_metrics.loc[a, "off_pass_epa"] - epa_metrics.loc[h, "def_pass_epa"])
            rush_edge = (epa_metrics.loc[h, "off_rush_epa"] - epa_metrics.loc[a, "def_rush_epa"]) - \
                        (epa_metrics.loc[a, "off_rush_epa"] - epa_metrics.loc[h, "def_rush_epa"])
            rows.append({
                "rating_diff": ratings[h] - ratings[a],
                "pass_edge": pass_edge,
                "rush_edge": rush_edge,
                "margin": row["home_score"] - row["away_score"],
                "home_win": int((row["home_score"] - row["away_score"]) > 0),
                "total_pts": row["home_score"] + row["away_score"],
                "weight": season_weight(row["season"], current_season, half_life)
            })

    df = pd.DataFrame(rows)
    X = df[["rating_diff", "pass_edge", "rush_edge"]].values
    w = df["weight"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression().fit(Xs, df["home_win"].values, sample_weight=w)
    reg_margin = Ridge(alpha=1.0).fit(Xs, df["margin"].values, sample_weight=w)
    reg_total = Ridge(alpha=1.0).fit(Xs, df["total_pts"].values, sample_weight=w)

    return {
        "scaler": scaler, "clf": clf, "reg_margin": reg_margin, "reg_total": reg_total,
        "margin_sigma": float(np.sqrt(np.average((df["margin"] - reg_margin.predict(Xs))**2, weights=w))),
        "total_sigma": float(np.sqrt(np.average((df["total_pts"] - reg_total.predict(Xs))**2, weights=w)))
    }

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

def run_monte_carlo(models, features, n_sims, spread, total):
    Xs = models["scaler"].transform([features])
    pred_margin = models["reg_margin"].predict(Xs)[0]
    pred_total = models["reg_total"].predict(Xs)[0]

    z1 = np.random.normal(0, 1, n_sims)
    z2 = np.random.normal(0, 1, n_sims)
    margin_sims = pred_margin + models["margin_sigma"] * z1
    total_sims = np.clip(pred_total + models["total_sigma"] * (0.15 * z1 + np.sqrt(1 - 0.15**2) * z2), 10, None)

    results = {
        "home_win_prob": float(np.mean(margin_sims > 0)),
        "cover_prob": float(np.mean(margin_sims > -spread)) if spread is not None else None,
        "over_prob": float(np.mean(total_sims > total)) if total is not None else None,
        "under_prob": float(np.mean(total_sims < total)) if total is not None else None
    }
    return results

# =========================================================
# MAIN APP EXECUTION
# =========================================================
current_season = get_current_nfl_season()

with st.spinner("Processing stats, power ratings, and historical simulations..."):
    schedules, pbp = load_data(LOOKBACK_YEARS, current_season)
    ratings = build_power_ratings(schedules, current_season, SEASON_HALF_LIFE)
    epa_metrics = build_epa_metrics(pbp, current_season, SEASON_HALF_LIFE)
    models = train_models(schedules, epa_metrics, ratings, current_season, half_life=SEASON_HALF_LIFE)

odds_data = fetch_nfl_odds(api_key)

if not api_key:
    st.info("💡 Enter your Odds API key in the sidebar to run live comparison calculations.")

# Fetch remaining upcoming games in current season
upcoming_games = schedules[
    (schedules["season"] == current_season) & 
    (schedules["home_score"].isna())
].sort_values("gameday")

if upcoming_games.empty:
    st.warning("No upcoming games found for the current season schedule.")
else:
    # Key odds lookup by standardized team abbreviations
    odds_lookup = {}
    if odds_data:
        for g in odds_data:
            a_abbr = to_abbr(g.get("away_team"))
            h_abbr = to_abbr(g.get("home_team"))
            odds_lookup[(a_abbr, h_abbr)] = g

    table_rows = []

    for _, row in upcoming_games.iterrows():
        home_abbr, away_abbr = row["home_team"], row["away_team"]

        if home_abbr not in ratings or away_abbr not in ratings or home_abbr not in epa_metrics.index or away_abbr not in epa_metrics.index:
            continue

        pass_edge = (epa_metrics.loc[home_abbr, "off_pass_epa"] - epa_metrics.loc[away_abbr, "def_pass_epa"]) - \
                    (epa_metrics.loc[away_abbr, "off_pass_epa"] - epa_metrics.loc[home_abbr, "def_pass_epa"])
        rush_edge = (epa_metrics.loc[home_abbr, "off_rush_epa"] - epa_metrics.loc[away_abbr, "def_rush_epa"]) - \
                    (epa_metrics.loc[away_abbr, "off_rush_epa"] - epa_metrics.loc[home_abbr, "def_rush_epa"])
        rating_diff = ratings[home_abbr] - ratings[away_abbr]

        features = [rating_diff, pass_edge, rush_edge]

        # Extract market lines
        game_odds = odds_lookup.get((away_abbr, home_abbr))
        market_spread, market_total, home_ml, away_ml = None, None, None, None

        if game_odds:
            for book in game_odds.get("bookmakers", []):
                for mk in book.get("markets", []):
                    if mk["key"] == "spreads" and market_spread is None:
                        market_spread = next((o.get("point") for o in mk["outcomes"] if to_abbr(o["name"]) == home_abbr), None)
                    if mk["key"] == "totals" and market_total is None:
                        market_total = next((o.get("point") for o in mk["outcomes"] if o["name"] == "Over"), None)
                    if mk["key"] == "h2h":
                        for o in mk["outcomes"]:
                            if to_abbr(o["name"]) == home_abbr and home_ml is None: home_ml = o.get("price")
                            if to_abbr(o["name"]) == away_abbr and away_ml is None: away_ml = o.get("price")

        sim = run_monte_carlo(models, features, N_SIMS, market_spread, market_total)

        # Calculate implied probabilities & Edges
        market_home_prob, market_away_prob = devig_two_way(american_to_prob(home_ml), american_to_prob(away_ml))
        
        edges = {}
        if market_home_prob:
            ml_edge = sim["home_win_prob"] - market_home_prob
            ml_label = f"Moneyline ({home_abbr} {int(home_ml):+d})" if home_ml is not None else f"Moneyline ({home_abbr})"
            edges[ml_label] = ml_edge if ml_edge > 0 else (sim["home_win_prob"] - 1 + market_away_prob)
            
        if sim["cover_prob"] is not None and market_spread is not None:
            edges[f"Spread ({home_abbr} {market_spread:+.1f})"] = sim["cover_prob"] - 0.524  # Standard -110 breakeven
            
        if sim["over_prob"] is not None and market_total is not None:
            edges[f"Total Over {market_total}"] = sim["over_prob"] - 0.524
            edges[f"Total Under {market_total}"] = sim["under_prob"] - 0.524

        # Select Best Bet Edge
        best_bet, best_edge_val = ("No market lines", 0.0)
        if edges:
            best_bet = max(edges, key=edges.get)
            best_edge_val = edges[best_bet]

        table_rows.append({
            "Date": str(row.get("gameday", ""))[:10],
            "Matchup": f"{away_abbr} @ {home_abbr}",
            "Model Win Prob": f"{sim['home_win_prob']*100:.1f}% ({home_abbr})",
            "Market Lines (Spd / Tot / ML)": f"{market_spread if market_spread is not None else '—'} | {market_total if market_total is not None else '—'} | {home_ml if home_ml is not None else '—'}",
            "Best Edge Market": best_bet,
            "Edge Value": f"{best_edge_val*100:+.1f}%" if edges else "—",
            "_sort": abs(best_edge_val)
        })

    if table_rows:
        df_display = pd.DataFrame(table_rows).sort_values("_sort", ascending=False).drop(columns=["_sort"])
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No upcoming games available to display.")
