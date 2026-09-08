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
    page_title="NFL Opportunity Scanner – Advanced",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 NFL Opportunity Scanner – Advanced Models")
st.caption("Week selector • Pass/Rush EPA • Recent Form • ML • Player Props")

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
st.sidebar.info("Player Props usually need a higher plan on The Odds API.")

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

def get_latest_available_season():
    """Return the most recent season that actually has data (currently 2025)."""
    # Try current calendar year first, then fall back
    for year in [2026, 2025, 2024]:
        try:
            # Quick test load
            test = nfl.load_schedules(seasons=[year])
            if test is not None and (hasattr(test, "height") and test.height > 0 or len(test) > 0):
                return year
        except Exception:
            continue
    return 2025  # safe fallback

@st.cache_data(ttl=3600 * 6)
def get_advanced_team_metrics(seasons=None):
    try:
        if seasons is None:
            latest = get_latest_available_season()
            seasons = [latest - 1, latest]

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

        return season.join(recent, how="outer").fillna(0)
    except Exception as e:
        st.warning(f"Metrics error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_schedules(seasons=None):
    try:
        if seasons is None:
            latest = get_latest_available_season()
            seasons = list(range(latest - 3, latest + 1))
        sched = nfl.load_schedules(seasons=seasons)
        return sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_current_schedule():
    try:
        latest = get_latest_available_season()
        sched = nfl.load_schedules(seasons=[latest])
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
        "player_pass_yds", "player_pass_tds", "player_rush_yds",
        "player_reception_yds", "player_receptions", "player_anytime_td",
        "player_pass_completions", "player_rush_tds", "player_reception_tds"
    ])
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american"
    }
    try:
        r = requests.get(url, params=params, timeout=25)
        if r.status_code == 200:
            return r.json()
        return {"error": f"Status {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

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

@st.cache_data(ttl=3600 * 4)
def load_recent_player_stats():
    try:
        latest = get_latest_available_season()
        stats = nfl.load_player_stats(seasons=[latest - 1, latest], summary_level="week")
        if hasattr(stats, "to_pandas"):
            stats = stats.to_pandas()
        return stats
    except Exception:
        return pd.DataFrame()

def get_player_recent_avg(player_stats, player_name, stat_col, last_n=8):
    if player_stats.empty or not player_name or stat_col not in player_stats.columns:
        return None, 0
    mask = (
        player_stats["player_display_name"].str.contains(player_name, case=False, na=False) |
        player_stats["player_name"].str.contains(player_name.split()[-1], case=False, na=False)
    )
    player_df = player_stats[mask].copy()
    if player_df.empty:
        return None, 0
    if "week" in player_df.columns and "season" in player_df.columns:
        player_df = player_df.sort_values(["season", "week"], ascending=False)
    recent = player_df.head(last_n)
    values = recent[stat_col].dropna()
    if len(values) == 0:
        return None, 0
    return float(values.mean()), len(values)

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

WEEK_OPTIONS = [f"Week {i}" for i in range(1, 19)] + [
    "Wild Card", "Divisional", "Conference Championship", "Super Bowl"
]

def filter_schedule_by_week(sched, selected):
    if sched.empty:
        return pd.DataFrame()
    if selected.startswith("Week"):
        week_num = int(selected.split()[1])
        return sched[sched["week"] == week_num].copy()
    elif selected == "Wild Card":
        return sched[sched["game_type"] == "WC"].copy()
    elif selected == "Divisional":
        return sched[sched["game_type"] == "DIV"].copy()
    elif selected == "Conference Championship":
        return sched[sched["game_type"] == "CON"].copy()
    elif selected == "Super Bowl":
        return sched[sched["game_type"] == "SB"].copy()
    return pd.DataFrame()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Opportunities",
    "📅 Games & Odds",
    "🎯 Player Props",
    "📊 Backtest",
    "🚀 Deploy"
])

# ========== TAB 1 ==========
with tab1:
    st.subheader("Advanced Opportunities")

    latest_season = get_latest_available_season()
    st.caption(f"Using latest available data: **{latest_season}** season (2026 data not published yet)")

    selected_week = st.selectbox(
        "Select Week / Round",
        options=WEEK_OPTIONS,
        index=0
    )

    with st.spinner(f"Loading {latest_season} schedule and advanced metrics..."):
        metrics = get_advanced_team_metrics()
        full_schedule = load_current_schedule()
        week_games = filter_schedule_by_week(full_schedule, selected_week)
        odds_data = fetch_nfl_odds(api_key) if api_key else None
        all_schedules = load_schedules()

    st.markdown(f"### {selected_week} ({latest_season})")

    if week_games.empty:
        st.info(f"No games found for {selected_week} in the {latest_season} season.")
    else:
        odds_lookup = {}
        if odds_data:
            for g in odds_data:
                key = (g.get("away_team"), g.get("home_team"))
                odds_lookup[key] = g

        opportunities = []

        for _, row in week_games.iterrows():
            home_full = row["home_team"]
            away_full = row["away_team"]
            home = to_abbr(home_full)
            away = to_abbr(away_full)
            gameday = str(row.get("gameday", ""))[:10]

            avg_spread = None
            game_odds = odds_lookup.get((away_full, home_full))
            if game_odds:
                spreads = []
                for book in game_odds.get("bookmakers", []):
                    for market in book.get("markets", []):
                        if market["key"] == "spreads":
                            for o in market["outcomes"]:
                                if o["name"] == home_full:
                                    spreads.append(o.get("point"))
                if spreads:
                    avg_spread = np.mean(spreads)

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

            home_rest = get_rest_days(all_schedules, home_full, gameday)
            away_rest = get_rest_days(all_schedules, away_full, gameday)
            rest_diff = home_rest - away_rest

            signals = []
            score = 0.0

            if pass_edge > 0.06:
                signals.append(f"Home Pass edge (+{pass_edge:.3f})")
                score += 2.4
            elif pass_edge < -0.06:
                signals.append(f"Away Pass edge ({pass_edge:.3f})")
                score += 2.2

            if rush_edge > 0.05:
                signals.append(f"Home Rush edge (+{rush_edge:.3f})")
                score += 1.8
            elif rush_edge < -0.05:
                signals.append(f"Away Rush edge ({rush_edge:.3f})")
                score += 1.6

            if recent_edge > 0.07:
                signals.append(f"Home Recent Form (+{recent_edge:.3f})")
                score += 2.8
            elif recent_edge < -0.07:
                signals.append(f"Away Recent Form ({recent_edge:.3f})")
                score += 2.6

            if rest_diff >= 3:
                signals.append(f"Home rest +{rest_diff}d")
                score += 1.2
            elif rest_diff <= -3:
                signals.append(f"Away rest {rest_diff}d")
                score += 1.1

            if avg_spread is not None and avg_spread > 1.5:
                signals.append("Home underdog")
                score += 1.1
            if avg_spread is not None and abs(avg_spread) >= 7:
                signals.append(f"Large spread ({avg_spread:+.1f})")
                score += 0.6

            if signals:
                opportunities.append({
                    "Game": f"{away_full} @ {home_full}",
                    "Date": gameday,
                    "Spread": f"{avg_spread:+.1f}" if avg_spread is not None else "—",
                    "Pass Edge": f"{pass_edge:+.3f}",
                    "Rush Edge": f"{rush_edge:+.3f}",
                    "Recent Form": f"{recent_edge:+.3f}",
                    "Signals": " • ".join(signals),
                    "Score": round(score, 1)
                })

        if opportunities:
            df = pd.DataFrame(opportunities).sort_values("Score", ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.success("Higher Score = stronger combination of Pass/Rush EPA + Recent Form + situational signals.")
        else:
            st.info("No strong signals for this week.")

# ========== TAB 2 ==========
with tab2:
    st.subheader("Upcoming Games & Current Lines")
    if odds_data:
        rows = []
        for g in odds_data:
            home = g["home_team"]
            away = g["away_team"]
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
            rows.append({
                "Away": away,
                "Home": home,
                "Kickoff": commence,
                "Spread": spread,
                "Total": total
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Enter an Odds API key in the sidebar to load live lines.")

# ========== TAB 3: PLAYER PROPS ==========
with tab3:
    st.subheader("Player Props Opportunity Scanner")
    st.write("Select a game to load available player props and score them against recent averages.")

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

        if st.button("Scan Player Props Opportunities", type="primary"):
            event_id = game_options[selected_game]
            with st.spinner("Fetching props + calculating recent averages..."):
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

                            if not player or line is None:
                                continue

                            avg, sample = get_player_recent_avg(player_stats, player, stat_col)
                            if avg is None:
                                continue

                            if side and "Over" in str(side):
                                edge = avg - line
                                preferred = "Over"
                            elif side and "Under" in str(side):
                                edge = line - avg
                                preferred = "Under"
                            else:
                                continue

                            score = 0.0
                            if abs(edge) >= 8:
                                score += 2.5
                            elif abs(edge) >= 4:
                                score += 1.6
                            elif abs(edge) >= 2:
                                score += 0.9
                            if sample >= 6:
                                score += 0.7
                            elif sample >= 3:
                                score += 0.3

                            if score > 0:
                                opportunities.append({
                                    "Player": player,
                                    "Market": market_label,
                                    "Side": preferred,
                                    "Line": line,
                                    "Recent Avg": round(avg, 1),
                                    "Edge": round(edge, 1),
                                    "Sample": sample,
                                    "Odds": odds,
                                    "Book": book_name,
                                    "Score": round(score, 1)
                                })

                if opportunities:
                    opp_df = pd.DataFrame(opportunities)
                    opp_df = opp_df.sort_values("Score", ascending=False)
                    opp_df = opp_df.drop_duplicates(subset=["Player", "Market", "Side"], keep="first")
                    st.success(f"Found {len(opp_df)} scored opportunities")
                    st.dataframe(opp_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No props could be matched to recent stats.")

# ========== TAB 4 ==========
with tab4:
    st.subheader("Improved Backtest – EPA Edge")
    min_edge = st.slider("Minimum EPA edge", 0.03, 0.20, 0.06, 0.01)
    seasons_back = st.multiselect("Seasons", [2022, 2023, 2024, 2025], default=[2023, 2024, 2025])

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            try:
                hist_sched = load_schedules(seasons=seasons_back)
                hist_epa = get_advanced_team_metrics(seasons=seasons_back)
                completed = hist_sched[hist_sched["result"].notna() & hist_sched["spread_line"].notna()]

                results = []
                for _, row in completed.iterrows():
                    home = to_abbr(row["home_team"])
                    away = to_abbr(row["away_team"])
                    if home not in hist_epa.index or away not in hist_epa.index:
                        continue
                    m = hist_epa.loc
                    recent_edge = (
                        (m[home, "off_pass_recent"] + m[home, "off_rush_recent"] - m[away, "def_pass_recent"] - m[away, "def_rush_recent"]) -
                        (m[away, "off_pass_recent"] + m[away, "off_rush_recent"] - m[home, "def_pass_recent"] - m[home, "def_rush_recent"])
                    ) / 2
                    if abs(recent_edge) < min_edge:
                        continue
                    covered = (row["result"] > row["spread_line"]) if recent_edge > 0 else (row["result"] < row["spread_line"])
                    results.append({"covered": covered})

                if results:
                    res_df = pd.DataFrame(results)
                    st.metric("ATS Win Rate", f"{res_df['covered'].mean():.1%}", delta=f"{len(res_df)} games")
                else:
                    st.warning("No qualifying games.")
            except Exception as e:
                st.error(str(e))

# ========== TAB 5 ==========
with tab5:
    st.subheader("Deploy / Update")
    st.markdown("""
    After making changes:
    1. Upload the new `app.py` to your GitHub repo
    2. Make sure `requirements.txt` includes streamlit, pandas, numpy, requests, nflreadpy, scikit-learn
    3. Go to share.streamlit.io → Reboot the app
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Full version • Auto-fallback to latest available season")
