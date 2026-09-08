import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import nflreadpy as nfl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NFL Opportunity Scanner",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"   # better for mobile
)

st.title("🏈 NFL Opportunity Scanner")
st.caption("Advanced models • Mobile optimized • Research tool only")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "The Odds API Key",
        type="password",
        help="Get a free key at https://the-odds-api.com"
    )
    st.markdown("---")
    st.info("Tip: On phone, tap the ☰ icon to open settings.")

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

        off_pass = pbp[pbp["play_type"] == "pass"].groupby("posteam")["epa"].mean().rename("off_pass_epa")
        off_rush = pbp[pbp["play_type"] == "run"].groupby("posteam")["epa"].mean().rename("off_rush_epa")
        def_pass = pbp[pbp["play_type"] == "pass"].groupby("defteam")["epa"].mean().rename("def_pass_epa")
        def_rush = pbp[pbp["play_type"] == "run"].groupby("defteam")["epa"].mean().rename("def_rush_epa")

        season = pd.concat([off_pass, off_rush, def_pass, def_rush], axis=1)

        if "week" in pbp.columns:
            max_week = pbp["week"].max()
            recent_pbp = pbp[pbp["week"] >= max(1, max_week - 5)]
        else:
            recent_pbp = pbp.tail(int(len(pbp) * 0.25))

        off_pass_r = recent_pbp[recent_pbp["play_type"] == "pass"].groupby("posteam")["epa"].mean().rename("off_pass_recent")
        off_rush_r = recent_pbp[recent_pbp["play_type"] == "run"].groupby("posteam")["epa"].mean().rename("off_rush_recent")
        def_pass_r = recent_pbp[recent_pbp["play_type"] == "pass"].groupby("defteam")["epa"].mean().rename("def_pass_recent")
        def_rush_r = recent_pbp[recent_pbp["play_type"] == "run"].groupby("defteam")["epa"].mean().rename("def_rush_recent")

        recent = pd.concat([off_pass_r, off_rush_r, def_pass_r, def_rush_r], axis=1)
        metrics = season.join(recent, how="outer").fillna(0)
        return metrics
    except Exception as e:
        st.warning(f"Metrics error: {e}")
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
        mask = (
            (schedules["home_team"] == team) |
            (schedules["away_team"] == team)
        ) & (schedules["gameday"] < str(game_date))
        team_games = schedules[mask].sort_values("gameday")
        if team_games.empty:
            return 7
        last = team_games.iloc[-1]["gameday"]
        return max((pd.to_datetime(game_date) - pd.to_datetime(last)).days, 0)
    except Exception:
        return 7

@st.cache_resource
def train_simple_ml_model():
    try:
        hist = load_schedules(seasons=[2022, 2023, 2024, 2025])
        metrics = get_advanced_team_metrics(seasons=[2022, 2023, 2024])
        if hist.empty or metrics.empty:
            return None, None

        completed = hist[hist["result"].notna() & hist["spread_line"].notna()].copy()
        rows = []
        for _, row in completed.iterrows():
            h = to_abbr(row["home_team"])
            a = to_abbr(row["away_team"])
            if h not in metrics.index or a not in metrics.index:
                continue
            m = metrics.loc
            pass_edge = (m[h, "off_pass_epa"] - m[a, "def_pass_epa"]) - (m[a, "off_pass_epa"] - m[h, "def_pass_epa"])
            rush_edge = (m[h, "off_rush_epa"] - m[a, "def_rush_epa"]) - (m[a, "off_rush_epa"] - m[h, "def_rush_epa"])
            recent_edge = (
                (m[h, "off_pass_recent"] + m[h, "off_rush_recent"] - m[a, "def_pass_recent"] - m[a, "def_rush_recent"]) -
                (m[a, "off_pass_recent"] + m[a, "off_rush_recent"] - m[h, "def_pass_recent"] - m[h, "def_rush_recent"])
            ) / 2
            covered = 1 if row["result"] > row["spread_line"] else 0
            rows.append({
                "pass_edge": pass_edge,
                "rush_edge": rush_edge,
                "recent_edge": recent_edge,
                "spread": row["spread_line"],
                "covered": covered
            })

        if len(rows) < 100:
            return None, None

        df = pd.DataFrame(rows)
        X = df[["pass_edge", "rush_edge", "recent_edge", "spread"]]
        y = df["covered"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=500)
        model.fit(X_scaled, y)
        return model, scaler
    except Exception:
        return None, None

# -----------------------------
# MAIN APP
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Opportunities", "📅 Games", "ℹ️ Info"])

with tab1:
    with st.spinner("Loading advanced models..."):
        metrics = get_advanced_team_metrics()
        schedules = load_schedules()
        odds_data = fetch_nfl_odds(api_key) if api_key else None
        ml_model, ml_scaler = train_simple_ml_model()

    opportunities = []

    if odds_data is not None and not metrics.empty:
        for game in odds_data:
            home_full = game.get("home_team")
            away_full = game.get("away_team")
            home = to_abbr(home_full)
            away = to_abbr(away_full)
            commence = game.get("commence_time", "")[:16].replace("T", " ")
            game_date = commence[:10] if commence else datetime.now().strftime("%Y-%m-%d")

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

            avg_spread = np.mean(spreads) if spreads else None

            h = home if home in metrics.index else home_full
            a = away if away in metrics.index else away_full
            if h not in metrics.index or a not in metrics.index:
                continue

            m = metrics.loc
            pass_edge = (m[h, "off_pass_epa"] - m[a, "def_pass_epa"]) - (m[a, "off_pass_epa"] - m[h, "def_pass_epa"])
            rush_edge = (m[h, "off_rush_epa"] - m[a, "def_rush_epa"]) - (m[a, "off_rush_epa"] - m[h, "def_rush_epa"])
            recent_edge = (
                (m[h, "off_pass_recent"] + m[h, "off_rush_recent"] - m[a, "def_pass_recent"] - m[a, "def_rush_recent"]) -
                (m[a, "off_pass_recent"] + m[a, "off_rush_recent"] - m[h, "def_pass_recent"] - m[h, "def_rush_recent"])
            ) / 2

            home_rest = get_rest_days(schedules, home_full, game_date)
            away_rest = get_rest_days(schedules, away_full, game_date)
            rest_diff = home_rest - away_rest

            signals = []
            score = 0.0

            if pass_edge > 0.06:
                signals.append(f"Pass +{pass_edge:.3f}")
                score += 2.4
            elif pass_edge < -0.06:
                signals.append(f"Pass {pass_edge:.3f}")
                score += 2.2

            if rush_edge > 0.05:
                signals.append(f"Rush +{rush_edge:.3f}")
                score += 1.8
            elif rush_edge < -0.05:
                signals.append(f"Rush {rush_edge:.3f}")
                score += 1.6

            if recent_edge > 0.07:
                signals.append(f"Form +{recent_edge:.3f}")
                score += 2.8
            elif recent_edge < -0.07:
                signals.append(f"Form {recent_edge:.3f}")
                score += 2.6

            if rest_diff >= 3:
                signals.append(f"Rest +{rest_diff}d")
                score += 1.2
            elif rest_diff <= -3:
                signals.append(f"Rest {rest_diff}d")
                score += 1.1

            if avg_spread is not None and avg_spread > 1.5:
                signals.append("Home Dog")
                score += 1.1

            ml_prob = None
            if ml_model is not None and ml_scaler is not None and avg_spread is not None:
                try:
                    features = np.array([[pass_edge, rush_edge, recent_edge, avg_spread]])
                    features_scaled = ml_scaler.transform(features)
                    ml_prob = ml_model.predict_proba(features_scaled)[0][1]
                    if ml_prob >= 0.58:
                        signals.append(f"ML {ml_prob:.0%}")
                        score += 1.8
                    elif ml_prob <= 0.42:
                        signals.append(f"ML {1-ml_prob:.0%}")
                        score += 1.7
                except Exception:
                    pass

            if score >= 2.0:  # only show meaningful opportunities
                opportunities.append({
                    "Game": f"{away_full.split()[-1]} @ {home_full.split()[-1]}",
                    "Full": f"{away_full} @ {home_full}",
                    "Spread": avg_spread,
                    "Pass": pass_edge,
                    "Rush": rush_edge,
                    "Form": recent_edge,
                    "ML": ml_prob,
                    "Signals": " • ".join(signals),
                    "Score": round(score, 1)
                })

    if opportunities:
        df = pd.DataFrame(opportunities).sort_values("Score", ascending=False).reset_index(drop=True)

        # ===== TOP METRICS (mobile friendly) =====
        top = df.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Top Score", f"{top['Score']}")
        col2.metric("Top Game", top["Game"])
        col3.metric("Signals", len(top["Signals"].split("•")))

        st.markdown("---")

        # ===== CHART 1: Score Ranking =====
        fig1 = px.bar(
            df.head(10),
            x="Score",
            y="Game",
            orientation="h",
            title="Top Opportunity Scores",
            color="Score",
            color_continuous_scale="Tealgrn"
        )
        fig1.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis={"categoryorder": "total ascending"}
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ===== CHART 2: Pass vs Rush Edge =====
        fig2 = px.scatter(
            df,
            x="Pass",
            y="Rush",
            size="Score",
            color="Score",
            hover_name="Full",
            title="Pass Edge vs Rush Edge",
            color_continuous_scale="Viridis"
        )
        fig2.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Full Ranked List")

        # Clean display table
        display_df = df[["Game", "Spread", "Pass", "Rush", "Form", "Score", "Signals"]].copy()
        display_df["Spread"] = display_df["Spread"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "—")
        display_df["Pass"] = display_df["Pass"].apply(lambda x: f"{x:+.3f}")
        display_df["Rush"] = display_df["Rush"].apply(lambda x: f"{x:+.3f}")
        display_df["Form"] = display_df["Form"].apply(lambda x: f"{x:+.3f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )

    else:
        if not api_key:
            st.warning("👉 Open the sidebar (☰) and paste your Odds API key.")
        else:
            st.info("No strong opportunities right now. Check back closer to game time.")

with tab2:
    st.subheader("Upcoming Games")
    if odds_data:
        rows = []
        for g in odds_data:
            home = g["home_team"]
            away = g["away_team"]
            commence = g.get("commence_time", "")[:16].replace("T", " ")
            spread = "—"
            for book in g.get("bookmakers", [])[:1]:
                for m in book.get("markets", []):
                    if m["key"] == "spreads":
                        for o in m["outcomes"]:
                            if o["name"] == home:
                                spread = f"{o.get('point', 0):+.1f}"
            rows.append({
                "Matchup": f"{away.split()[-1]} @ {home.split()[-1]}",
                "Kickoff": commence,
                "Spread": spread
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Enter API key to see games.")

with tab3:
    st.markdown("""
    ### How to read the scanner
    - **Score** → Higher is stronger combination of signals  
    - **Pass / Rush** → EPA matchup edges  
    - **Form** → Recent performance edge (last ~6 weeks)  
    - **ML** → Simple machine learning cover probability  

    This is a research tool only. Never bet more than you can afford to lose.
    """)

st.caption("Mobile optimized • Advanced models active")
