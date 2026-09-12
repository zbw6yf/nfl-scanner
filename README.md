# TAIL ME Sports — Modular Structure

NFL betting research / opportunity scanner (Streamlit).

## New layout

```
nfl-scanner/
├── app.py                 # Entry point + tab orchestration (still contains most UI + domain logic)
├── byoa.py                # Build-Your-Own-Algorithm module (already modular)
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── constants.py       # Team maps, stadiums, colors, divisions, feature flags
├── utils/
│   ├── __init__.py
│   ├── teams.py           # to_abbr, full_name, divisional, TZ, travel
│   ├── dates.py           # kickoff formatting, week estimation
│   ├── odds_utils.py      # implied totals, vig, edge, CLV, American odds
│   └── features.py        # feature flags, user_tier, Pro CTA
├── ui/
│   ├── __init__.py
│   └── theme.py           # CSS injection, logo, conf pills, freshness stamps
├── persistence/
│   ├── __init__.py
│   └── storage.py         # API key, signal history, bet log file I/O
├── data/                  # (next extraction target)
│   └── __init__.py
└── models/                # (next extraction target)
    └── (empty for now)
```

## What was extracted

| Module | Contents |
|--------|----------|
| `config/constants.py` | TEAM_NAME_TO_ABBR, ABBR_TO_FULL, STADIUM_COORDS, TEAM_TZ, DIVISIONS, TEAM_COLORS, CONF_COLORS, DEFAULT_FEATURE_FLAGS |
| `utils/teams.py` | to_abbr, full_name, expand_team, is_divisional, timezone_diff, travel_direction, normalize_team_abbr |
| `utils/dates.py` | format_kickoff, format_schedule_kickoff, estimate_week_from_date, current_nfl_week |
| `utils/odds_utils.py` | implied_team_totals, american_to_implied_prob, remove_vig_two_way, compute_edge, CLV helpers, american_profit |
| `utils/features.py` | feature_enabled, user_tier, require_pro, stripe helpers, render_upgrade_cta |
| `ui/theme.py` | inject_theme_css, get_logo_data_uri, conf_pill, stamp_now / stamp_text / last_update_caption |
| `persistence/storage.py` | API key + signal history + bet log load/save |

## How to run

```bash
cd nfl-scanner
pip install -r requirements.txt
streamlit run app.py
```

You still need an Odds API key (enter it in the sidebar). Optional: `tailme_logo.png` next to `app.py`.

## Next extraction steps (recommended order)

1. **data/loaders.py** — `load_schedules*`, `fetch_nfl_odds`, injury / depth / weather fetchers  
2. **data/epa.py** — `get_team_epa`, `get_team_pace`, `get_recent_form`, success metrics  
3. **models/scoring.py** — `confidence_grade`, Monte Carlo, logistic training  
4. **ui/tabs/** — one file per major tab (Homepage, Big Board, Games & Odds, …)

The current `app.py` still contains the original implementations of the heavier functions so the app keeps working while you migrate. Once a function is moved, delete the local copy and import from the new module.

## Notes

- Paths for persisted files (API key, signal history, bet log) are relative to the project root.
- Feature flags can be overridden via `st.secrets["features"]`.
- `byoa.py` was already clean and is left unchanged.
