import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import nflreadpy as nfl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NFL Opportunity Scanner – ML + Monte Carlo",
    page_icon="🏈",
    layout="wide"
)
st.title("🏈 NFL Betting Opportunity Scanner")
st.caption("EPA + Rules + Rest + ML + Monte Carlo. Research tool only. Not financial advice.")

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
        off = pbp.groupby("posteam")["epa"].agg(off_epa="mean").reset_index().rename(columns={"posteam": "team"})
        deff = pbp.groupby("defteam")["epa"].agg(def_epa="mean").reset_index().rename(columns={"defteam": "team"})
        team_epa = off.merge(deff, on="team", how="outer")
        return team_epa.set_index("team")
    except Exception:
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

def fetch_player_props(api_key: str, event_id: str):
    if not api_key or not event_id:
        return None
    markets = ",".join([
        "player_pass_yds",
        "player_pass_tds",
        "player_rush_yds",
        "player_reception_yds",
        "player_receptions",
        "player_anytime_td",
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

# -----------------------------
# MACHINE LEARNING + MONTE CARLO HELPERS
# -----------------------------
@st.cache_data(ttl=3600 * 12)
def prepare_historical_features(seasons):
    """Build feature matrix for training an ATS model."""
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
            result = float(row["result"])  # home margin
            covered = 1 if result > spread else 0  # home covers

            # rest (approximate – full rest history is expensive, use simple prior)
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
                "epa_edge": epa_edge,
                "spread": spread,
                "rest_diff": rest_diff,
                "home_off": home_off,
                "home_def": home_def,
                "away_off": away_off,
                "away_def": away_def,
                "abs_spread": abs(spread),
                "total_line": total_line,
                "home_covered": covered,
                "margin": result
            })

        df = pd.DataFrame(rows)
        if len(df) < 100:
            return None, None
        return df, epa
    except Exception:
        return None, None

@st.cache_resource(ttl=3600 * 12)
def train_ats_model(seasons):
    """Train a simple logistic regression pipeline for home ATS cover probability."""
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
    noise_std=11.5  # roughly historical residual std of NFL margins
):
    """
    Simulate final margins and totals.
    Expected margin is driven by EPA differential (scaled).
    """
    # Rough conversion: EPA edge of ~0.10 ≈ 3–4 points of expected margin
    expected_margin = (home_off - away_def - (away_off - home_def)) * 35.0
    # slight home field residual already partially in EPA; keep small extra
    expected_margin += 1.2

    # simulate margins
    sim_margins = np.random.normal(loc=expected_margin, scale=noise_std, size=n_sims)

    # simulate totals (simple independent-ish model)
    expected_total = 44.0 + (home_off + away_off - home_def - away_def) * 22.0
    sim_totals = np.random.normal(loc=expected_total, scale=13.5, size=n_sims)

    home_cover_prob = np.mean(sim_margins > spread)
    away_cover_prob = 1.0 - home_cover_prob
    over_prob = np.mean(sim_totals > total_line) if total_line else 0.5
    under_prob = 1.0 - over_prob

    # expected value style edge assuming -110
    # positive = value on that side
    home_ev = home_cover_prob * 100/110 - (1 - home_cover_prob)
    away_ev = away_cover_prob * 100/110 - (1 - away_cover_prob)

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

def american_to_implied(odds):
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Opportunities (ML + MC)",
    "📅 Games & Odds",
    "🎯 Player Props",
    "📊 Backtest",
    "🚀 Deploy"
])

# ========== TAB 1: OPPORTUNITIES ==========
with tab1:
    st.subheader("Ranked Game Opportunities – EPA + ML + Monte Carlo")
    st.caption("Score blends original rule signals, ML cover probability, and Monte Carlo edge.")

    with st.spinner("Loading data & training model..."):
        team_epa = get_team_epa()
        schedules = load_schedules()
        odds_data = fetch_nfl_odds(api_key) if api_key else None

        # Train on recent seasons (exclude current if still early)
        current_season = nfl.get_current_season()
        train_seasons = list(range(current_season - 4, current_season))
        model_bundle = train_ats_model(train_seasons)

    opportunities = []
    if odds_data and not team_epa.empty:
        model = model_bundle[0] if model_bundle else None
        feature_cols = model_bundle[1] if model_bundle else None

        for game in odds_data:
            home = game.get("home_team")
            away = game.get("away_team")
            commence = game.get("commence_time", "")[:16].replace("T", " ")
            game_date = commence[:10] if commence else datetime.now().strftime("%Y-%m-%d")

            spreads, totals = [], []
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market["key"] == "spreads":
                        for o in market["outcomes"]:
                            if o["name"] == home:
                                spreads.append(o.get("point"))
                    elif market["key"] == "totals":
                        for o in market["outcomes"]:
                            if o["name"] == "Over":
                                totals.append(o.get("point"))

            avg_spread = float(np.mean(spreads)) if spreads else None
            avg_total = float(np.mean(totals)) if totals else 45.0

            if home not in team_epa.index or away not in team_epa.index:
                continue

            home_off = team_epa.loc[home, "off_epa"]
            home_def = team_epa.loc[home, "def_epa"]
            away_off = team_epa.loc[away, "off_epa"]
            away_def = team_epa.loc[away, "def_epa"]

            epa_edge = (home_off - away_def) - (away_off - home_def)
            home_rest = get_rest_days(schedules, home, game_date)
            away_rest = get_rest_days(schedules, away, game_date)
            rest_diff = home_rest - away_rest

            # ---------- Original rule signals ----------
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

            # ---------- Machine Learning probability ----------
            ml_home_cover = 0.5
            if model is not None and avg_spread is not None:
                feat = pd.DataFrame([{
                    "epa_edge": epa_edge,
                    "spread": avg_spread,
                    "rest_diff": rest_diff,
                    "home_off": home_off,
                    "home_def": home_def,
                    "away_off": away_off,
                    "away_def": away_def,
                    "abs_spread": abs(avg_spread),
                    "total_line": avg_total
                }])[feature_cols]
                ml_home_cover = float(model.predict_proba(feat)[0, 1])

            # ---------- Monte Carlo ----------
            mc = monte_carlo_game(
                home_off, home_def, away_off, away_def,
                avg_spread if avg_spread is not None else 0.0,
                avg_total,
                n_sims=n_simulations
            )

            # Combined score
            # ML contribution: distance from 50%
            ml_edge = abs(ml_home_cover - 0.5) * 4.0
            # MC contribution: max of the two EV sides (scaled)
            mc_edge = max(mc["home_ev"], mc["away_ev"]) * 8.0
            # prefer the side the models agree on
            agreement_bonus = 0.0
            preferred_side = "Home" if (ml_home_cover > 0.5 and mc["home_cover_prob"] > 0.52) or \
                                      (ml_home_cover < 0.5 and mc["home_cover_prob"] < 0.48) else "Split"
            if preferred_side != "Split":
                agreement_bonus = 1.5

            total_score = rule_score + ml_edge + mc_edge + agreement_bonus

            # Recommendation
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

            if signals or total_score > 2.5:
                opportunities.append({
                    "Game": f"{away} @ {home}",
                    "Kickoff": commence,
                    "Spread": f"{avg_spread:+.1f}" if avg_spread is not None else "—",
                    "Total": f"{avg_total:.1f}" if avg_total is not None else "—",
                    "EPA Edge": f"{epa_edge:+.3f}",
                    "ML Home Cover %": f"{ml_home_cover*100:.1f}%",
                    "MC Home Cover %": f"{mc['home_cover_prob']*100:.1f}%",
                    "MC Home EV": f"{mc['home_ev']:+.3f}",
                    "MC Away EV": f"{mc['away_ev']:+.3f}",
                    "Recommendation": rec,
                    "Signals": " • ".join(signals) if signals else "—",
                    "Score": round(total_score, 2)
                })

    if opportunities:
        df = pd.DataFrame(opportunities).sort_values("Score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### How the new score is built")
        st.markdown("""
        - **Rule score**: original EPA / rest / underdog / large-spread / high-total points  
        - **ML edge**: logistic regression trained on recent seasons predicting home ATS cover  
        - **Monte Carlo edge**: thousands of simulated margins & totals → cover probabilities & EV at -110  
        - **Agreement bonus**: when ML and Monte Carlo point the same direction  
        """)
    else:
        st.info("Add your API key or wait for stronger signals. Model needs historical data to train.")

# ========== TAB 2: GAMES ==========
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
        st.info("Enter API key to load games.")

# ========== TAB 3: PLAYER PROPS ==========
with tab3:
    st.subheader("Player Props Scanner")
    st.write("Select a game to load available player props (Passing Yards, Rushing Yards, Receptions, TDs, etc.)")
    if not api_key:
        st.warning("Enter your Odds API key in the sidebar first.")
    elif not odds_data:
        st.info("No games available right now.")
    else:
        game_options = {
            f"{g['away_team']} @ {g['home_team']}": g["id"]
            for g in odds_data
        }
        selected_game = st.selectbox("Choose a game", options=list(game_options.keys()))
        if st.button("Load Player Props for this game", type="primary"):
            event_id = game_options[selected_game]
            with st.spinner("Fetching player props... (this can take 10–20 seconds)"):
                props_data = fetch_player_props(api_key, event_id)
            if props_data is None:
                st.error("Failed to fetch props.")
            elif "error" in props_data:
                st.error(f"API returned an error: {props_data.get('error')}")
                st.write("Most free/basic plans do not include full NFL player props. You may need to upgrade your Odds API plan.")
            else:
                rows = []
                for book in props_data.get("bookmakers", []):
                    book_name = book.get("title", book.get("key"))
                    for market in book.get("markets", []):
                        market_key = market.get("key", "")
                        for outcome in market.get("outcomes", []):
                            rows.append({
                                "Book": book_name,
                                "Market": market_key.replace("player_", "").replace("_", " ").title(),
                                "Player": outcome.get("description", outcome.get("name", "")),
                                "Side": outcome.get("name"),
                                "Line": outcome.get("point"),
                                "Odds": outcome.get("price")
                            })
                if rows:
                    props_df = pd.DataFrame(rows)
                    st.success(f"Found {len(props_df)} prop lines")
                    st.dataframe(props_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No player props returned for this game on your current API plan.")

# ========== TAB 4: BACKTEST ==========
with tab4:
    st.subheader("Improved Backtest – EPA + ML")
    st.write("Trains the ML model on earlier seasons and evaluates ATS performance on later seasons.")

    min_edge = st.slider("Minimum EPA edge (rule filter)", 0.03, 0.20, 0.05, 0.01)
    seasons_back = st.multiselect(
        "Evaluation seasons",
        [2021, 2022, 2023, 2024, 2025],
        default=[2023, 2024, 2025]
    )
    train_on = st.multiselect(
        "Train seasons (must be before eval)",
        [2019, 2020, 2021, 2022, 2023, 2024],
        default=[2020, 2021, 2022]
    )

    if st.button("Run Backtest"):
        with st.spinner("Training model & running backtest..."):
            try:
                # Train
                model_bundle = train_ats_model(train_on)
                if model_bundle is None:
                    st.error("Not enough historical data to train.")
                else:
                    model, feature_cols = model_bundle
                    hist_sched = load_schedules(seasons=seasons_back)
                    hist_epa = get_team_epa(seasons=seasons_back)
                    completed = hist_sched[
                        hist_sched["result"].notna() &
                        hist_sched["spread_line"].notna()
                    ]

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

                        # rest approximation
                        rest_diff = 0
                        total_line = row.get("total_line", 45.0)
                        if pd.isna(total_line):
                            total_line = 45.0

                        feat = pd.DataFrame([{
                            "epa_edge": epa_edge,
                            "spread": spread,
                            "rest_diff": rest_diff,
                            "home_off": home_off,
                            "home_def": home_def,
                            "away_off": away_off,
                            "away_def": away_def,
                            "abs_spread": abs(spread),
                            "total_line": total_line
                        }])[feature_cols]

                        ml_prob = float(model.predict_proba(feat)[0, 1])

                        # Decision rules
                        side = None
                        if epa_edge >= min_edge and ml_prob > 0.52:
                            side = "Home"
                            covered = result > spread
                        elif epa_edge <= -min_edge and ml_prob < 0.48:
                            side = "Away"
                            covered = result < spread

                        if side:
                            results.append({
                                "side": side,
                                "covered": covered,
                                "ml_prob": ml_prob,
                                "epa_edge": epa_edge
                            })

                    if results:
                        res_df = pd.DataFrame(results)
                        win_rate = res_df["covered"].mean()
                        st.metric("ATS Win Rate (ML + EPA filter)", f"{win_rate:.1%}", delta=f"{len(res_df)} bets")
                        st.write(f"Home bets: {(res_df['side']=='Home').sum()} | Away bets: {(res_df['side']=='Away').sum()}")
                        st.dataframe(
                            res_df.describe()[["ml_prob", "epa_edge"]].T,
                            use_container_width=True
                        )
                    else:
                        st.warning("No qualifying games under current filters.")
            except Exception as e:
                st.error(str(e))

# ========== TAB 5: DEPLOY ==========
with tab5:
    st.subheader("Deploy / Update your app")
    st.markdown(""")
    After making changes:
    1. Upload the new `app.py` to your GitHub repo
    2. Make sure `requirements.txt` contains:
