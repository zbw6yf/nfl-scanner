"""
BYOA — Build Your Own Algorithm
================================
Standalone module for the TAIL ME Sports Streamlit app.

Implements:
1. Exact mapping from Big Board opportunity row keys
2. Self-contained feature rebuild (no Big Board visit required)
3. Clean modular API so app.py only needs a thin tab wrapper

Usage in app.py:
    from byoa import render_byoa_tab, features_dict_for_board_row

    # When appending Big Board opportunities, attach structured features:
    opportunity["_features"] = features_dict_for_board_row(...)

    # In tabs:
    with tab10:
        render_byoa_tab(
            api_key=api_key,
            n_simulations=n_simulations,
            form_window=form_window,
            # optional callables from app.py for self-contained rebuild:
            build_upcoming_games_fn=build_upcoming_games,
            get_schedules_fn=...,
            ...
        )
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Factor catalog
# ---------------------------------------------------------------------------

BYOA_FACTORS: Dict[str, Dict[str, Any]] = {
    "epa_edge": {
        "label": "EPA edge (home − away matchup)",
        "help": "Positive favors home ATS. Typical range ≈ -0.3 to +0.3.",
        "default_weight": 2.0,
        "default_enabled": True,
    },
    "form_margin_diff": {
        "label": "Recent form margin (home − away)",
        "help": "Avg point differential over recent games. Positive favors home.",
        "default_weight": 1.5,
        "default_enabled": True,
    },
    "form_epa_diff": {
        "label": "Recent form EPA (home − away)",
        "help": "Recent EPA differential. Positive favors home.",
        "default_weight": 1.2,
        "default_enabled": True,
    },
    "success_rate_edge": {
        "label": "Success rate edge (home − away matchup)",
        "help": "Home off success − Away def success, minus the reverse. Typical range ≈ -0.08 to +0.08. Positive favors home.",
        "default_weight": 1.6,
        "default_enabled": True,
    },
    "explosive_rate_edge": {
        "label": "Explosive play rate edge (home − away)",
        "help": "Differential in chunk-play rate (EPA ≥ 1.0). Typical range ≈ -0.05 to +0.05. Positive favors home.",
        "default_weight": 1.3,
        "default_enabled": True,
    },
    "redzone_td_edge": {
        "label": "Red-zone TD rate edge (home − away)",
        "help": "Red-zone touchdown rate differential (off vs opp def). Typical range ≈ -0.15 to +0.15. Positive favors home.",
        "default_weight": 1.4,
        "default_enabled": True,
    },
    "rest_diff": {
        "label": "Rest advantage (home − away days)",
        "help": "Positive means home has more rest.",
        "default_weight": 1.0,
        "default_enabled": True,
    },
    "spread": {
        "label": "Spread (home perspective)",
        "help": "Positive = home underdog. Useful as dog-bias factor.",
        "default_weight": 0.8,
        "default_enabled": False,
    },
    "abs_spread": {
        "label": "Absolute spread size",
        "help": "Large numbers = blowout spots. Use carefully.",
        "default_weight": 0.3,
        "default_enabled": False,
    },
    "total_line": {
        "label": "Game total",
        "help": "Centered around 45. Higher can lean Over with pace.",
        "default_weight": 0.5,
        "default_enabled": False,
    },
    "pace_vs_avg": {
        "label": "Combined pace vs league avg",
        "help": "Positive = faster game → mild Over lean in totals mode.",
        "default_weight": 0.7,
        "default_enabled": True,
    },
    "weather_under_bias": {
        "label": "Weather under bias",
        "help": "Higher wind/precip/cold pressure toward Under.",
        "default_weight": 1.0,
        "default_enabled": True,
    },
    "home_imp": {
        "label": "Home implied team total",
        "help": "From spread + total when available.",
        "default_weight": 0.6,
        "default_enabled": False,
    },
    "away_imp": {
        "label": "Away implied team total",
        "help": "From spread + total when available.",
        "default_weight": 0.6,
        "default_enabled": False,
    },
    "tz_diff": {
        "label": "Timezone disadvantage (away)",
        "help": "Hours of TZ travel for away team. Higher can favor home.",
        "default_weight": 0.5,
        "default_enabled": False,
    },
    "divisional": {
        "label": "Divisional game flag",
        "help": "1 if same division else 0.",
        "default_weight": 0.4,
        "default_enabled": False,
    },
    "model_home_prob": {
        "label": "Model home cover probability",
        "help": "From your logistic / blend model when present on board rows.",
        "default_weight": 1.0,
        "default_enabled": False,
    },
    "edge_pct": {
        "label": "Model vs market edge %",
        "help": "Positive = model likes the recommended side more than market.",
        "default_weight": 0.8,
        "default_enabled": False,
    },
}

PRESET_PATH = Path(__file__).resolve().parent / "byoa_presets.json"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def byoa_default_config() -> Dict[str, Any]:
    factors = {}
    for k, meta in BYOA_FACTORS.items():
        factors[k] = {
            "enabled": bool(meta["default_enabled"]),
            "weight": float(meta["default_weight"]),
            "invert": False,
        }
    return {
        "name": "My Algorithm",
        "factors": factors,
        "min_abs_score": 1.25,
        "prefer_side": "Auto",       # Auto | Home | Away
        "market": "ATS",             # ATS | Total
        "total_side_mode": "Auto",   # Auto | Over | Under
    }


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str):
            s = x.strip().replace("%", "").replace("—", "").replace("–", "")
            if s in ("", "nan", "None", "N/A", "-"):
                return default
            # handle "+3.5", "-7.0"
            return float(s)
        if isinstance(x, (float, int, np.floating, np.integer)):
            if isinstance(x, float) and np.isnan(x):
                return default
            return float(x)
        return float(x)
    except Exception:
        return default


def _parse_signed_number(text: Any) -> Optional[float]:
    if text is None:
        return None
    if isinstance(text, (int, float, np.floating, np.integer)):
        return float(text)
    s = str(text).strip()
    if not s or s in ("—", "–", "-", "N/A", "None", "nan"):
        return None
    m = re.search(r"([+-]?\d+\.?\d*)", s.replace("%", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Option 1 — exact Big Board key mapping
# ---------------------------------------------------------------------------

def features_from_board_row(o: Dict[str, Any]) -> Dict[str, float]:
    """
    Map a Big Board opportunity row (display + hidden keys) into BYOA factors.

    Known Big Board keys (from app.py):
      Display: Week, Game, Kickoff, Roof, Weather, Spread, Total,
               Home Imp, Away Imp, EPA Edge, Form Δ, Pace, TZ Diff, Div,
               Model %, Market %, Edge %, ML Home %, MC Home %, MC Over %,
               Recommendation, Confidence, Signals, Score
      Hidden:  _model_prob, _market_prob, _edge_pct, _spread, _total,
               _home, _away, _gameday
      Optional structured: _features (preferred when present)
    """
    if isinstance(o.get("_features"), dict) and o["_features"]:
        return normalize_features(o["_features"])

    # Hidden numerics first (most reliable)
    spread = o.get("_spread")
    if spread is None:
        spread = _parse_signed_number(o.get("Spread"))

    total = o.get("_total")
    if total is None:
        total = _parse_signed_number(o.get("Total"))

    epa_edge = _parse_signed_number(o.get("EPA Edge"))
    form_margin = _parse_signed_number(o.get("Form Δ") or o.get("Form Delta"))
    pace = _parse_signed_number(o.get("Pace"))
    home_imp = _parse_signed_number(o.get("Home Imp"))
    away_imp = _parse_signed_number(o.get("Away Imp"))
    tz_raw = o.get("TZ Diff")
    tz_diff = _parse_signed_number(tz_raw) if tz_raw is not None else 0.0
    div_val = o.get("Div")
    if isinstance(div_val, str):
        divisional = 1.0 if div_val.strip().lower() in ("yes", "y", "true", "1") else 0.0
    else:
        divisional = 1.0 if div_val else 0.0

    model_home = o.get("_model_prob")
    if model_home is None:
        model_home = _parse_signed_number(o.get("Model %"))
        if model_home is not None and model_home > 1.5:
            model_home = model_home / 100.0

    edge_pct = o.get("_edge_pct")
    if edge_pct is None:
        edge_pct = _parse_signed_number(o.get("Edge %"))

    # Signals text fallbacks
    signals = str(o.get("Signals") or "")
    if epa_edge is None:
        m = re.search(r"(Home|Away)\s+EPA\s*([+-]?\d+\.?\d*)", signals, re.I)
        if m:
            val = float(m.group(2))
            epa_edge = val if m.group(1).lower() == "home" else -abs(val)
    rest_diff = None
    m = re.search(r"(Home|Away)\s+rest\s*([+-]?\d+)", signals, re.I)
    if m:
        val = float(m.group(2))
        rest_diff = val if m.group(1).lower() == "home" else -abs(val)
    form_epa = None
    m = re.search(r"(Home|Away)\s+form EPA\s*([+-]?\d+\.?\d*)", signals, re.I)
    if m:
        val = float(m.group(2))
        form_epa = val if m.group(1).lower() == "home" else -abs(val)
    weather_bias = 0.0
    if re.search(r"wind|rain|snow|cold|weather", signals, re.I):
        weather_bias = 1.0
        m = re.search(r"rule[_\s]?pts.*?(\d+\.?\d*)", signals, re.I)
        # soft default when weather signal present
        weather_bias = 1.0

    # Pace vs avg: board stores raw combined pace; approximate vs 65
    pace_vs_avg = None
    if pace is not None:
        pace_vs_avg = pace - 65.0

    # Parse optional efficiency edges from Signals text when not on the row
    success_edge = _parse_signed_number(o.get("Success Edge"))
    explosive_edge = _parse_signed_number(o.get("Explosive Edge"))
    redzone_edge = _parse_signed_number(o.get("RZ TD Edge"))
    if success_edge is None:
        m = re.search(r"(Home|Away)\s+success\s*([+-]?\d+\.?\d*)", signals, re.I)
        if m:
            val = float(m.group(2))
            success_edge = val if m.group(1).lower() == "home" else -abs(val)
    if explosive_edge is None:
        m = re.search(r"(Home|Away)\s+explosive\s*([+-]?\d+\.?\d*)", signals, re.I)
        if m:
            val = float(m.group(2))
            explosive_edge = val if m.group(1).lower() == "home" else -abs(val)
    if redzone_edge is None:
        m = re.search(r"(Home|Away)\s+RZ(?:\s*TD)?\s*([+-]?\d+\.?\d*)", signals, re.I)
        if m:
            val = float(m.group(2))
            redzone_edge = val if m.group(1).lower() == "home" else -abs(val)

    raw = {
        "epa_edge": epa_edge if epa_edge is not None else 0.0,
        "form_margin_diff": form_margin if form_margin is not None else 0.0,
        "form_epa_diff": form_epa if form_epa is not None else 0.0,
        "success_rate_edge": success_edge if success_edge is not None else 0.0,
        "explosive_rate_edge": explosive_edge if explosive_edge is not None else 0.0,
        "redzone_td_edge": redzone_edge if redzone_edge is not None else 0.0,
        "rest_diff": rest_diff if rest_diff is not None else 0.0,
        "spread": spread if spread is not None else 0.0,
        "abs_spread": abs(spread) if spread is not None else 0.0,
        "total_line": total if total is not None else 45.0,
        "pace_vs_avg": pace_vs_avg if pace_vs_avg is not None else 0.0,
        "weather_under_bias": weather_bias,
        "home_imp": home_imp if home_imp is not None else 0.0,
        "away_imp": away_imp if away_imp is not None else 0.0,
        "tz_diff": tz_diff if tz_diff is not None else 0.0,
        "divisional": divisional,
        "model_home_prob": model_home if model_home is not None else 0.5,
        "edge_pct": edge_pct if edge_pct is not None else 0.0,
    }
    return normalize_features(raw)


def features_dict_for_board_row(
    *,
    epa_edge: float = 0.0,
    form_margin_diff: float = 0.0,
    form_epa_diff: float = 0.0,
    success_rate_edge: float = 0.0,
    explosive_rate_edge: float = 0.0,
    redzone_td_edge: float = 0.0,
    rest_diff: float = 0.0,
    avg_spread: Optional[float] = None,
    avg_total: Optional[float] = None,
    pace_vs_avg: float = 0.0,
    weather_under_bias: float = 0.0,
    home_imp: float = 0.0,
    away_imp: float = 0.0,
    tz_diff: float = 0.0,
    divisional: bool = False,
    model_home_prob: float = 0.5,
    edge_pct: float = 0.0,
) -> Dict[str, float]:
    """
    Call this inside the Big Board loop when building each opportunity:

        opportunity["_features"] = features_dict_for_board_row(
            epa_edge=epa_edge,
            form_margin_diff=form_margin_diff,
            success_rate_edge=success_rate_edge,
            explosive_rate_edge=explosive_rate_edge,
            redzone_td_edge=redzone_td_edge,
            ...
        )
    """
    spread = float(avg_spread) if avg_spread is not None else 0.0
    total = float(avg_total) if avg_total is not None else 45.0
    return normalize_features({
        "epa_edge": epa_edge,
        "form_margin_diff": form_margin_diff,
        "form_epa_diff": form_epa_diff,
        "success_rate_edge": success_rate_edge,
        "explosive_rate_edge": explosive_rate_edge,
        "redzone_td_edge": redzone_td_edge,
        "rest_diff": rest_diff,
        "spread": spread,
        "abs_spread": abs(spread),
        "total_line": total,
        "pace_vs_avg": pace_vs_avg,
        "weather_under_bias": weather_under_bias,
        "home_imp": home_imp,
        "away_imp": away_imp,
        "tz_diff": tz_diff,
        "divisional": 1.0 if divisional else 0.0,
        "model_home_prob": model_home_prob,
        "edge_pct": edge_pct,
    })


def normalize_features(raw: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k in BYOA_FACTORS:
        out[k] = _safe_float(raw.get(k), 0.0)
    # ensure abs_spread consistent
    out["abs_spread"] = abs(_safe_float(raw.get("spread", out.get("abs_spread", 0.0))))
    if "total_line" in out and out["total_line"] == 0.0 and raw.get("total_line") is None:
        out["total_line"] = 45.0
    return out


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def _scale_feature(key: str, raw: float) -> float:
    """Bring heterogeneous features onto roughly comparable scales."""
    if key in ("epa_edge", "form_epa_diff"):
        return raw * 10.0
    # Success / explosive / RZ rates are already small fractions
    if key == "success_rate_edge":
        return raw * 25.0          # ±0.08 → ±2.0
    if key == "explosive_rate_edge":
        return raw * 30.0          # ±0.05 → ±1.5
    if key == "redzone_td_edge":
        return raw * 12.0          # ±0.15 → ±1.8
    if key in ("form_margin_diff", "rest_diff", "pace_vs_avg", "spread", "tz_diff"):
        return raw / 3.0
    if key == "abs_spread":
        return raw / 7.0
    if key == "total_line":
        return (raw - 45.0) / 5.0
    if key in ("home_imp", "away_imp"):
        return (raw - 22.0) / 5.0
    if key == "weather_under_bias":
        return -abs(raw)  # pushes Under in totals mode
    if key == "divisional":
        return raw  # 0/1
    if key == "model_home_prob":
        return (raw - 0.5) * 10.0
    if key == "edge_pct":
        return raw / 5.0
    return raw


def score_game(features: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Weighted linear score.
    ATS: positive => Home lean, negative => Away lean
    Total: positive => Over lean, negative => Under lean
    """
    market = cfg.get("market", "ATS")
    contribs: List[Dict[str, Any]] = []
    score = 0.0

    for key, fcfg in (cfg.get("factors") or {}).items():
        if not fcfg.get("enabled"):
            continue
        if key not in BYOA_FACTORS:
            continue
        raw = _safe_float(features.get(key), 0.0)
        w = _safe_float(fcfg.get("weight"), 0.0)
        if fcfg.get("invert"):
            w = -w

        scaled = _scale_feature(key, raw)

        # ATS-specific polarity tweaks
        if market == "ATS":
            if key == "away_imp":
                scaled = -scaled
            if key == "weather_under_bias":
                scaled *= 0.25  # muted on ATS
            if key == "total_line":
                scaled *= 0.35
        else:
            # Totals: model_home_prob / rest less relevant by default
            if key in ("model_home_prob", "rest_diff", "tz_diff", "divisional"):
                scaled *= 0.35
            if key == "epa_edge":
                scaled *= 0.5

        part = w * scaled
        score += part
        contribs.append({
            "factor": key,
            "label": BYOA_FACTORS[key]["label"],
            "raw": round(raw, 4),
            "scaled": round(scaled, 4),
            "weight": round(w, 3),
            "contrib": round(part, 4),
        })

    contribs.sort(key=lambda x: abs(x["contrib"]), reverse=True)
    min_abs = _safe_float(cfg.get("min_abs_score"), 1.25)

    if market == "ATS":
        prefer = cfg.get("prefer_side", "Auto")
        if prefer == "Home":
            side = "Home"
        elif prefer == "Away":
            side = "Away"
        else:
            side = "Home" if score >= 0 else "Away"
        if abs(score) < min_abs:
            rec, conf = "No strong lean", "F"
        else:
            rec = f"{side} ATS"
            conf = _conf_from_abs(abs(score))
    else:
        mode = cfg.get("total_side_mode", "Auto")
        if mode == "Over":
            side = "Over"
        elif mode == "Under":
            side = "Under"
        else:
            side = "Over" if score >= 0 else "Under"
        if abs(score) < min_abs:
            rec, conf = "No strong lean", "F"
        else:
            rec = side
            conf = _conf_from_abs(abs(score))

    return {
        "score": float(score),
        "abs_score": float(abs(score)),
        "recommendation": rec,
        "confidence": conf,
        "contributions": contribs,
    }


def _conf_from_abs(a: float) -> str:
    if a >= 4.0:
        return "A"
    if a >= 3.0:
        return "B"
    if a >= 2.0:
        return "C"
    if a >= 1.25:
        return "D"
    return "F"


def run_byoa_on_rows(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> pd.DataFrame:
    out = []
    for o in rows or []:
        feats = features_from_board_row(o)
        scored = score_game(feats, cfg)
        home = o.get("_home") or o.get("Home") or ""
        away = o.get("_away") or o.get("Away") or ""
        game = o.get("Game") or (f"{away} @ {home}" if (away or home) else "—")
        out.append({
            "Week": o.get("Week", "—"),
            "Game": game,
            "Kickoff": o.get("Kickoff", "—"),
            "Spread": o.get("Spread", o.get("_spread", "—")),
            "Total": o.get("Total", o.get("_total", "—")),
            "BYOA Score": round(scored["score"], 3),
            "Recommendation": scored["recommendation"],
            "Confidence": scored["confidence"],
            "Top drivers": ", ".join(
                f"{c['factor']} ({c['contrib']:+.2f})" for c in scored["contributions"][:3]
            ),
            "_contribs": scored["contributions"],
            "_features": feats,
            "_home": home,
            "_away": away,
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    conf_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    df["_cr"] = df["Confidence"].map(lambda x: conf_rank.get(str(x), 9))
    df["_abs"] = df["BYOA Score"].abs()
    df = df.sort_values(["_cr", "_abs"], ascending=[True, False]).drop(columns=["_cr", "_abs"])
    return df


# ---------------------------------------------------------------------------
# Option 2 — self-contained feature rebuild (no Big Board required)
# ---------------------------------------------------------------------------

def rebuild_feature_rows(
    *,
    api_key: str = "",
    form_window: int = 6,
    # Injected callables from app.py (keeps module decoupled from Streamlit data loaders)
    load_schedules: Optional[Callable[[], pd.DataFrame]] = None,
    load_odds: Optional[Callable[[str], Optional[List]]] = None,
    build_upcoming: Optional[Callable[..., List[Dict]]] = None,
    get_team_epa: Optional[Callable[..., pd.DataFrame]] = None,
    get_team_pace: Optional[Callable[..., pd.DataFrame]] = None,
    get_team_success_metrics: Optional[Callable[..., pd.DataFrame]] = None,
    get_recent_form: Optional[Callable[..., Dict]] = None,
    rest_differential: Optional[Callable[..., int]] = None,
    is_divisional: Optional[Callable[..., bool]] = None,
    timezone_diff: Optional[Callable[..., int]] = None,
    weather_cache_builder: Optional[Callable[..., Dict]] = None,
    weather_adjustments: Optional[Callable[..., Dict]] = None,
    implied_team_totals: Optional[Callable[..., Tuple[float, float]]] = None,
    estimate_week: Optional[Callable[..., Optional[int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build lightweight feature rows for upcoming games without requiring
    the user to open The Big Board first.

    Pass the real functions from app.py. Missing callables → graceful empty list.
    """
    required = [load_schedules, build_upcoming, get_team_epa]
    if any(fn is None for fn in required):
        return []

    try:
        schedules = load_schedules()
        if schedules is None or getattr(schedules, "empty", True):
            return []
    except Exception:
        return []

    odds_data = None
    if load_odds and api_key:
        try:
            odds_data = load_odds(api_key)
        except Exception:
            odds_data = None

    try:
        upcoming = build_upcoming(schedules, odds_data)
    except Exception:
        return []
    if not upcoming:
        return []

    try:
        team_epa = get_team_epa() if get_team_epa else pd.DataFrame()
    except Exception:
        team_epa = pd.DataFrame()
    try:
        team_pace = get_team_pace() if get_team_pace else pd.DataFrame()
    except Exception:
        team_pace = pd.DataFrame()
    try:
        recent_form = get_recent_form(form_window) if get_recent_form else {}
    except Exception:
        recent_form = {}

    league_avg_pace = 65.0
    if isinstance(team_pace, pd.DataFrame) and not team_pace.empty and "plays_per_game" in team_pace.columns:
        try:
            league_avg_pace = float(team_pace["plays_per_game"].mean())
        except Exception:
            pass

    weather_cache: Dict[str, Dict] = {}
    if weather_cache_builder:
        try:
            weather_cache = weather_cache_builder(upcoming) or {}
        except Exception:
            weather_cache = {}

    rows: List[Dict[str, Any]] = []
    for g in upcoming:
        try:
            home = str(g.get("home") or "").upper()
            away = str(g.get("away") or "").upper()
            home_full = g.get("home_full") or home
            away_full = g.get("away_full") or away
            game_date = str(g.get("gameday") or "")[:10]
            commence = g.get("kickoff") or ""
            commence_raw = g.get("commence_raw") or ""
            roof = str(g.get("roof") or "outdoors")
            avg_spread = g.get("avg_spread")
            avg_total = g.get("avg_total") if g.get("avg_total") is not None else 45.0
            week = g.get("week")
            if week is None and estimate_week and game_date:
                try:
                    week = estimate_week(game_date)
                except Exception:
                    week = None

            # EPA edge
            epa_edge = 0.0
            if isinstance(team_epa, pd.DataFrame) and not team_epa.empty and home in team_epa.index and away in team_epa.index:
                home_off = float(team_epa.loc[home, "off_epa"])
                home_def = float(team_epa.loc[home, "def_epa"])
                away_off = float(team_epa.loc[away, "off_epa"])
                away_def = float(team_epa.loc[away, "def_epa"])
                epa_edge = (home_off - away_def) - (away_off - home_def)

            home_form = recent_form.get(home, {"form_margin": 0.0, "form_epa": 0.0, "n": 0})
            away_form = recent_form.get(away, {"form_margin": 0.0, "form_epa": 0.0, "n": 0})
            form_margin_diff = float(home_form.get("form_margin", 0.0)) - float(away_form.get("form_margin", 0.0))
            form_epa_diff = float(home_form.get("form_epa", 0.0)) - float(away_form.get("form_epa", 0.0))

            rest_diff = 0
            if rest_differential:
                try:
                    rest_diff = int(rest_differential(schedules, home, away, game_date, week=week))
                except Exception:
                    rest_diff = 0

            home_pace = league_avg_pace
            away_pace = league_avg_pace
            if isinstance(team_pace, pd.DataFrame) and not team_pace.empty:
                if home in team_pace.index and "plays_per_game" in team_pace.columns:
                    home_pace = float(team_pace.loc[home, "plays_per_game"])
                if away in team_pace.index and "plays_per_game" in team_pace.columns:
                    away_pace = float(team_pace.loc[away, "plays_per_game"])
            combined_pace = (home_pace + away_pace) / 2.0
            pace_vs_avg = combined_pace - league_avg_pace

            tz = 0
            if timezone_diff:
                try:
                    tz = int(timezone_diff(home, away))
                except Exception:
                    tz = 0
            div_flag = False
            if is_divisional:
                try:
                    div_flag = bool(is_divisional(home, away))
                except Exception:
                    div_flag = False

            # weather
            weather_under_bias = 0.0
            wx_str = "—"
            if weather_adjustments:
                wx = {"temp_f": 70.0, "wind_mph": 5.0, "precip_prob": 10.0, "roof": roof, "source": "missing"}
                # try cache key patterns used in app
                for key in (
                    f"{home}_{commence_raw}",
                    f"{home}_{game_date}",
                    commence_raw,
                    game_date,
                ):
                    if key in weather_cache:
                        wx = weather_cache[key]
                        break
                # also try values if cache is keyed differently
                if weather_cache and wx.get("source") == "missing":
                    for v in weather_cache.values():
                        if isinstance(v, dict):
                            wx = v
                            break
                try:
                    adj = weather_adjustments(roof, wx) or {}
                    weather_under_bias = float(adj.get("under_bias") or adj.get("rule_pts") or 0.0)
                    wx_str = str(adj.get("label") or wx_str)
                except Exception:
                    pass

            if implied_team_totals and avg_spread is not None:
                try:
                    home_imp, away_imp = implied_team_totals(avg_spread, avg_total)
                except Exception:
                    home_imp = away_imp = float(avg_total) / 2.0
            else:
                home_imp = away_imp = float(avg_total or 45.0) / 2.0

            # Efficiency edges (success / explosive / RZ) — optional if caller provides metrics
            success_rate_edge = 0.0
            explosive_rate_edge = 0.0
            redzone_td_edge = 0.0
            team_success = None
            if get_team_success_metrics:
                try:
                    team_success = get_team_success_metrics()
                except Exception:
                    team_success = None
            if isinstance(team_success, pd.DataFrame) and not team_success.empty:
                if home in team_success.index and away in team_success.index:
                    try:
                        h_sr = float(team_success.loc[home, "off_success"]) - float(team_success.loc[away, "def_success"])
                        a_sr = float(team_success.loc[away, "off_success"]) - float(team_success.loc[home, "def_success"])
                        success_rate_edge = h_sr - a_sr
                    except Exception:
                        pass
                    try:
                        h_exp = float(team_success.loc[home, "off_explosive"]) - float(team_success.loc[away, "def_explosive"])
                        a_exp = float(team_success.loc[away, "off_explosive"]) - float(team_success.loc[home, "def_explosive"])
                        explosive_rate_edge = h_exp - a_exp
                    except Exception:
                        pass
                    # Red-zone columns optional
                    if "off_rz_td" in team_success.columns and "def_rz_td" in team_success.columns:
                        try:
                            h_rz = float(team_success.loc[home, "off_rz_td"]) - float(team_success.loc[away, "def_rz_td"])
                            a_rz = float(team_success.loc[away, "off_rz_td"]) - float(team_success.loc[home, "def_rz_td"])
                            redzone_td_edge = h_rz - a_rz
                        except Exception:
                            pass

            feats = features_dict_for_board_row(
                epa_edge=epa_edge,
                form_margin_diff=form_margin_diff,
                form_epa_diff=form_epa_diff,
                success_rate_edge=success_rate_edge,
                explosive_rate_edge=explosive_rate_edge,
                redzone_td_edge=redzone_td_edge,
                rest_diff=float(rest_diff),
                avg_spread=float(avg_spread) if avg_spread is not None else None,
                avg_total=float(avg_total) if avg_total is not None else 45.0,
                pace_vs_avg=pace_vs_avg,
                weather_under_bias=weather_under_bias,
                home_imp=float(home_imp),
                away_imp=float(away_imp),
                tz_diff=float(tz),
                divisional=div_flag,
            )

            rows.append({
                "Week": week if week is not None else "—",
                "Game": f"{away_full} @ {home_full}",
                "Kickoff": commence,
                "Spread": f"{avg_spread:+.1f}" if avg_spread is not None else "—",
                "Total": f"{float(avg_total):.1f}",
                "Home Imp": f"{float(home_imp):.1f}",
                "Away Imp": f"{float(away_imp):.1f}",
                "EPA Edge": f"{epa_edge:+.3f}",
                "Form Δ": f"{form_margin_diff:+.1f}",
                "Pace": f"{combined_pace:.1f}",
                "TZ Diff": f"{tz}h" if tz else "0",
                "Div": "Yes" if div_flag else "No",
                "Weather": wx_str,
                "_spread": avg_spread,
                "_total": avg_total,
                "_home": home,
                "_away": away,
                "_gameday": game_date,
                "_features": feats,
            })
        except Exception:
            continue
    return rows


# ---------------------------------------------------------------------------
# Presets persistence
# ---------------------------------------------------------------------------

def load_presets() -> Dict[str, Any]:
    presets = dict(st.session_state.get("byoa_presets") or {})
    try:
        if PRESET_PATH.exists():
            disk = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
            if isinstance(disk, dict):
                presets.update(disk)
    except Exception:
        pass
    return presets


def save_preset(name: str, cfg: Dict[str, Any]) -> None:
    presets = load_presets()
    presets[name] = cfg
    st.session_state["byoa_presets"] = presets
    try:
        PRESET_PATH.write_text(json.dumps(presets, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Option 3 — Streamlit tab UI (modular)
# ---------------------------------------------------------------------------

def render_byoa_tab(
    *,
    api_key: str = "",
    n_simulations: int = 8000,
    form_window: int = 6,
    load_schedules: Optional[Callable] = None,
    load_odds: Optional[Callable] = None,
    build_upcoming: Optional[Callable] = None,
    get_team_epa: Optional[Callable] = None,
    get_team_pace: Optional[Callable] = None,
    get_team_success_metrics: Optional[Callable] = None,
    get_recent_form: Optional[Callable] = None,
    rest_differential: Optional[Callable] = None,
    is_divisional: Optional[Callable] = None,
    timezone_diff: Optional[Callable] = None,
    weather_cache_builder: Optional[Callable] = None,
    weather_adjustments: Optional[Callable] = None,
    implied_team_totals: Optional[Callable] = None,
    estimate_week: Optional[Callable] = None,
) -> None:
    """Render the full BYOA tab — clearer step-by-step layout."""

    # ---- Hero ----
    st.markdown(
        """
<div style="
  background: linear-gradient(135deg, #0b1220 0%, #1e3a5f 55%, #1d4ed8 100%);
  border-radius: 16px; padding: 1.25rem 1.4rem; margin-bottom: 0.85rem;
  border: 1px solid rgba(255,255,255,0.08); color: #f8fafc;">
  <div style="font-size:1.4rem;font-weight:700;margin-bottom:0.35rem;">🧪 Build Your Own Algorithm</div>
  <div style="color:#cbd5e1;font-size:0.95rem;line-height:1.5;max-width:52rem;">
    Three steps: <b>choose a style</b> → <b>pick factors</b> → <b>run on this week’s games</b>.
    Transparent weighted score — research only, not betting advice.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Session config + migration ----
    if "byoa_cfg" not in st.session_state:
        st.session_state["byoa_cfg"] = byoa_default_config()
    cfg = st.session_state["byoa_cfg"]
    if not isinstance(cfg.get("factors"), dict):
        cfg["factors"] = {}
    for _k, _meta in BYOA_FACTORS.items():
        if _k not in cfg["factors"]:
            cfg["factors"][_k] = {
                "enabled": bool(_meta["default_enabled"]),
                "weight": float(_meta["default_weight"]),
                "invert": False,
            }
    st.session_state["byoa_cfg"] = cfg
    cfg = st.session_state["byoa_cfg"]

    FACTOR_GROUPS = {
        "Core matchup": [
            "epa_edge", "form_margin_diff", "form_epa_diff", "rest_diff",
            "success_rate_edge", "explosive_rate_edge", "redzone_td_edge",
        ],
        "Market lines": [
            "spread", "abs_spread", "total_line",
            "home_imp", "away_imp", "edge_pct", "model_home_prob",
        ],
        "Context": [
            "pace_vs_avg", "weather_under_bias", "tz_diff", "divisional",
        ],
    }

    def _apply_preset(name: str, market: str, enabled: tuple, weights: Optional[Dict[str, float]] = None):
        c = byoa_default_config()
        c["name"] = name
        c["market"] = market
        for k, f in c["factors"].items():
            f["enabled"] = k in enabled
            if weights and k in weights:
                f["weight"] = float(weights[k])
        st.session_state["byoa_cfg"] = c
        for _k in list(st.session_state.keys()):
            if isinstance(_k, str) and _k.startswith(("byoa_en_", "byoa_w_", "byoa_inv_")):
                del st.session_state[_k]
        st.rerun()

    # ========== STEP 1: Style ==========
    st.markdown("### Step 1 — Choose a starting point")
    st.caption("Pick a preset, or keep your current setup and edit factors below.")

    pcols = st.columns(4)
    with pcols[0]:
        if st.button("⭐ Balanced defaults", use_container_width=True, help="EPA, form, rest, pace, weather"):
            st.session_state["byoa_cfg"] = byoa_default_config()
            for _k in list(st.session_state.keys()):
                if isinstance(_k, str) and _k.startswith(("byoa_en_", "byoa_w_", "byoa_inv_")):
                    del st.session_state[_k]
            st.rerun()
    with pcols[1]:
        if st.button("📈 EPA + efficiency", use_container_width=True, help="Matchup quality & recent form"):
            _apply_preset(
                "EPA + efficiency",
                "ATS",
                (
                    "epa_edge", "form_margin_diff", "form_epa_diff", "rest_diff",
                    "success_rate_edge", "explosive_rate_edge", "redzone_td_edge",
                ),
                {"epa_edge": 2.5, "form_margin_diff": 1.8, "form_epa_diff": 1.8,
                 "success_rate_edge": 1.8, "explosive_rate_edge": 1.4, "redzone_td_edge": 1.4},
            )
    with pcols[2]:
        if st.button("🐕 Underdog lean", use_container_width=True, help="Home dogs, rest, travel"):
            _apply_preset(
                "Underdog lean",
                "ATS",
                ("spread", "rest_diff", "epa_edge", "tz_diff"),
                {"spread": 1.6, "rest_diff": 1.2, "epa_edge": 1.5, "tz_diff": 0.8},
            )
    with pcols[3]:
        if st.button("🌧️ Totals / weather", use_container_width=True, help="Pace, total, weather for O/U"):
            _apply_preset(
                "Totals / weather",
                "Total",
                ("total_line", "pace_vs_avg", "weather_under_bias", "home_imp", "away_imp"),
                {"weather_under_bias": 1.5, "pace_vs_avg": 1.2, "total_line": 0.8},
            )

    # Saved presets row
    presets = load_presets()
    with st.expander("Saved presets", expanded=False):
        pc1, pc2, pc3 = st.columns([1.2, 1.2, 1])
        with pc1:
            cfg["name"] = st.text_input("Name for this algorithm", value=cfg.get("name", "My Algorithm"))
            if st.button("💾 Save current setup", use_container_width=True):
                save_preset(cfg["name"], deepcopy(cfg))
                st.success(f"Saved “{cfg['name']}”")
        with pc2:
            names = sorted(presets.keys())
            if names:
                pick = st.selectbox("Load a saved preset", names, key="byoa_preset_pick")
                if st.button("Load selected", use_container_width=True) and pick:
                    st.session_state["byoa_cfg"] = deepcopy(presets[pick])
                    for _k in list(st.session_state.keys()):
                        if isinstance(_k, str) and _k.startswith(("byoa_en_", "byoa_w_", "byoa_inv_")):
                            del st.session_state[_k]
                    st.rerun()
            else:
                st.caption("No saved presets yet.")
        with pc3:
            st.caption("Saving stores factor weights and market settings for later.")

    st.markdown("---")

    # ========== STEP 2: Configure ==========
    st.markdown("### Step 2 — Configure market & factors")

    setup_l, setup_r = st.columns([1, 1.15], gap="large")

    with setup_l:
        st.markdown("#### Market")
        cfg["market"] = st.radio(
            "What are you scoring?",
            ["ATS", "Total"],
            horizontal=True,
            index=0 if cfg.get("market", "ATS") == "ATS" else 1,
            help="ATS = against the spread. Total = over/under.",
            key="byoa_market_radio",
        )
        if cfg["market"] == "ATS":
            opts = ["Auto", "Home", "Away"]
            cfg["prefer_side"] = st.selectbox(
                "Prefer side (optional bias)",
                opts,
                index=opts.index(cfg.get("prefer_side", "Auto")) if cfg.get("prefer_side") in opts else 0,
                help="Auto lets the score decide. Home/Away nudges only when close.",
            )
        else:
            opts = ["Auto", "Over", "Under"]
            cfg["total_side_mode"] = st.selectbox(
                "Prefer total side (optional)",
                opts,
                index=opts.index(cfg.get("total_side_mode", "Auto")) if cfg.get("total_side_mode") in opts else 0,
            )

        cfg["min_abs_score"] = st.slider(
            "Minimum strength to show a lean",
            0.0, 5.0, float(cfg.get("min_abs_score", 1.25)), 0.05,
            help="Weaker scores show as “No strong lean”. Raise this to be pickier.",
        )

        # Active summary
        n_on = sum(1 for f in cfg["factors"].values() if f.get("enabled"))
        st.markdown(
            f"""
<div style="background:rgba(29,78,216,0.12);border:1px solid rgba(96,165,250,0.35);
border-radius:10px;padding:0.7rem 0.9rem;margin-top:0.5rem;">
  <div style="font-weight:600;color:#93c5fd;">Active setup</div>
  <div style="color:#e2e8f0;font-size:0.92rem;margin-top:0.25rem;">
    <b>{cfg.get('name', 'My Algorithm')}</b> · {cfg.get('market', 'ATS')} ·
    {n_on} factor{'s' if n_on != 1 else ''} on · min score {float(cfg.get('min_abs_score', 1.25)):.2f}
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with setup_r:
        st.markdown("#### Factors")
        st.caption("Toggle a factor on, then set how much it matters (weight). Invert flips direction.")

        fr1, fr2 = st.columns(2)
        with fr1:
            if st.button("Reset factor weights", key="byoa_reset_factors", use_container_width=True):
                st.session_state["byoa_cfg"] = byoa_default_config()
                for _k in list(st.session_state.keys()):
                    if isinstance(_k, str) and _k.startswith(("byoa_en_", "byoa_w_", "byoa_inv_")):
                        del st.session_state[_k]
                st.rerun()
        with fr2:
            only_on = st.toggle("Show only enabled", value=False, key="byoa_only_enabled")

        enabled_labels = []
        for group_name, keys in FACTOR_GROUPS.items():
            with st.expander(f"{group_name}", expanded=(group_name == "Core matchup")):
                for key in keys:
                    meta = BYOA_FACTORS[key]
                    fcfg = cfg["factors"].setdefault(
                        key,
                        {"enabled": meta["default_enabled"], "weight": meta["default_weight"], "invert": False},
                    )
                    if only_on and not fcfg.get("enabled"):
                        # still keep key in cfg; just hide UI
                        continue

                    on = st.checkbox(
                        meta["label"],
                        value=bool(fcfg.get("enabled")),
                        key=f"byoa_en_{key}",
                        help=meta.get("help", ""),
                    )
                    fcfg["enabled"] = on
                    if on:
                        wcol, icol = st.columns([3, 1])
                        with wcol:
                            fcfg["weight"] = st.slider(
                                "Weight",
                                min_value=0.0,
                                max_value=5.0,
                                value=float(abs(fcfg.get("weight", meta["default_weight"]))),
                                step=0.1,
                                key=f"byoa_w_{key}",
                                label_visibility="collapsed",
                            )
                        with icol:
                            fcfg["invert"] = st.checkbox(
                                "Invert",
                                value=bool(fcfg.get("invert")),
                                key=f"byoa_inv_{key}",
                            )
                        inv = " ↓" if fcfg.get("invert") else ""
                        enabled_labels.append(f"{meta['label'].split('(')[0].strip()}{inv} ×{fcfg['weight']:.1f}")
                    cfg["factors"][key] = fcfg

        # orphan factors safety
        grouped = {k for keys in FACTOR_GROUPS.values() for k in keys}
        for key, meta in BYOA_FACTORS.items():
            if key in grouped:
                continue
            cfg["factors"].setdefault(
                key,
                {"enabled": meta["default_enabled"], "weight": meta["default_weight"], "invert": False},
            )

        if enabled_labels:
            st.success("**On:** " + " · ".join(enabled_labels[:12]) + (" …" if len(enabled_labels) > 12 else ""))
        else:
            st.warning("Turn on at least one factor to score games.")

    st.session_state["byoa_cfg"] = cfg

    st.markdown("---")

    # ========== STEP 3: Load & run ==========
    st.markdown("### Step 3 — Load games & run")

    board_opps = st.session_state.get("bb_opportunities") or []
    standalone_opps = st.session_state.get("byoa_standalone_rows") or []

    src_l, src_r = st.columns([1.2, 1])
    with src_l:
        source = st.radio(
            "Where should game data come from?",
            [
                "Big Board (recommended — fastest)",
                "Rebuild from schedule (standalone)",
            ],
            index=0 if board_opps else (0 if not standalone_opps else 1),
            key="byoa_source_v2",
            help="Big Board reuses rows already built in The Big Board tab.",
        )
    with src_r:
        m1, m2 = st.columns(2)
        m1.metric("Big Board games", len(board_opps))
        m2.metric("Standalone games", len(standalone_opps))

    if source.startswith("Big Board"):
        base_rows = board_opps
        if not base_rows:
            st.info(
                "No Big Board data in this session yet. "
                "Open **The Big Board** once (and wait for it to finish), then come back here — "
                "or choose **Rebuild from schedule**."
            )
    else:
        base_rows = standalone_opps
        can_rebuild = all(fn is not None for fn in (load_schedules, build_upcoming, get_team_epa))
        if not can_rebuild:
            st.warning("Standalone rebuild isn’t fully connected. Use Big Board data instead.")
        if st.button(
            "🔄 Rebuild feature rows",
            type="secondary",
            disabled=not can_rebuild,
            use_container_width=True,
        ):
            with st.spinner("Building feature rows…"):
                standalone_opps = rebuild_feature_rows(
                    api_key=api_key,
                    form_window=form_window,
                    load_schedules=load_schedules,
                    load_odds=load_odds,
                    build_upcoming=build_upcoming,
                    get_team_epa=get_team_epa,
                    get_team_pace=get_team_pace,
                    get_team_success_metrics=get_team_success_metrics,
                    get_recent_form=get_recent_form,
                    rest_differential=rest_differential,
                    is_divisional=is_divisional,
                    timezone_diff=timezone_diff,
                    weather_cache_builder=weather_cache_builder,
                    weather_adjustments=weather_adjustments,
                    implied_team_totals=implied_team_totals,
                    estimate_week=estimate_week,
                )
                st.session_state["byoa_standalone_rows"] = standalone_opps
                st.success(f"Built {len(standalone_opps)} games")
            base_rows = st.session_state.get("byoa_standalone_rows") or []

    if base_rows:
        def _week_sort_key(w: str):
            s = str(w).strip()
            if s.isdigit():
                return (0, int(s))
            import re as _re
            m = _re.match(r"(\d+)", s)
            if m:
                return (0, int(m.group(1)))
            return (1, s)

        week_set = {str(r.get("Week") or "—") for r in base_rows}
        weeks = sorted(week_set, key=_week_sort_key)

        _cur = None
        try:
            if estimate_week:
                _cur = estimate_week(datetime.now().strftime("%Y-%m-%d"))
        except Exception:
            _cur = None
        if _cur is None:
            try:
                from utils.dates import current_nfl_week as _cnw
                _cur = _cnw()
            except Exception:
                _cur = None

        default_weeks = weeks
        if _cur is not None:
            cur_str = str(int(_cur))
            matches = [w for w in weeks if str(w).strip() == cur_str or str(w).strip().lstrip("0") == cur_str]
            if matches:
                default_weeks = matches
            else:
                numeric = [(w, int(str(w).strip())) for w in weeks if str(w).strip().isdigit()]
                le = [w for w, n in numeric if n <= int(_cur)]
                if le:
                    default_weeks = [max(le, key=lambda x: int(str(x).strip()))]

        week_pick = st.multiselect(
            "Weeks to include",
            weeks,
            default=default_weeks,
            key="byoa_weeks_v2",
        )
        filtered = [r for r in base_rows if str(r.get("Week") or "—") in set(week_pick)]

        run_c1, run_c2 = st.columns([2, 1])
        with run_c1:
            st.caption(f"**{len(filtered)}** games ready with your current factor setup.")
        with run_c2:
            run = st.button("▶  Run algorithm", type="primary", use_container_width=True)

        if run:
            if not filtered:
                st.warning("Select at least one week.")
            elif n_on == 0:
                st.warning("Enable at least one factor in Step 2.")
            else:
                with st.spinner("Scoring games…"):
                    result = run_byoa_on_rows(filtered, cfg)
                    st.session_state["byoa_result"] = result

        result = st.session_state.get("byoa_result")
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.markdown("### Results")
            leans = result[result["Recommendation"].astype(str) != "No strong lean"].copy()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Games scored", len(result))
            c2.metric("Strong leans", len(leans))
            top_conf = str(leans["Confidence"].iloc[0]) if len(leans) else "—"
            c3.metric("Top confidence", top_conf)
            c4.metric("Market", cfg.get("market", "ATS"))

            # Top lean cards
            if len(leans):
                st.markdown("#### Top leans")
                top_n = leans.head(3)
                cards = st.columns(min(3, len(top_n)))
                for i, (_, r) in enumerate(top_n.iterrows()):
                    with cards[i]:
                        rec = str(r.get("Recommendation", "—"))
                        conf = str(r.get("Confidence", "—"))
                        score = float(r.get("BYOA Score", 0) or 0)
                        game = str(r.get("Game", "—"))
                        st.markdown(
                            f"""
<div style="background:#0f172a;border:1px solid rgba(148,163,184,0.25);border-radius:12px;
padding:0.85rem 1rem;min-height:7.5rem;">
  <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.2rem;">#{i+1} · Conf {conf}</div>
  <div style="font-weight:700;font-size:1.05rem;color:#f8fafc;margin-bottom:0.35rem;">{rec}</div>
  <div style="color:#cbd5e1;font-size:0.88rem;line-height:1.35;">{game}</div>
  <div style="margin-top:0.45rem;color:#93c5fd;font-size:0.85rem;">Score {score:+.2f}</div>
</div>
                            """,
                            unsafe_allow_html=True,
                        )

            show = result.drop(
                columns=[c for c in ("_contribs", "_features", "_home", "_away") if c in result.columns]
            )
            with st.expander("Full results table", expanded=True):
                st.dataframe(show, use_container_width=True, hide_index=True, height=340)

            st.markdown("#### Inspect one game")
            labels = show["Game"].astype(str).tolist()
            choice = st.selectbox("Game", labels, key="byoa_inspect")
            row = result[result["Game"].astype(str) == choice].iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Recommendation", str(row["Recommendation"]))
            m2.metric("Confidence", str(row["Confidence"]))
            m3.metric("BYOA Score", f"{float(row['BYOA Score']):+.3f}")

            contribs = row.get("_contribs") or []
            if contribs:
                with st.expander("What drove this score?", expanded=True):
                    cdf = pd.DataFrame(contribs)
                    # Friendlier column order if present
                    prefer = [c for c in ("label", "factor", "raw", "scaled", "weight", "contrib") if c in cdf.columns]
                    if prefer:
                        cdf = cdf[prefer + [c for c in cdf.columns if c not in prefer]]
                    st.dataframe(cdf, use_container_width=True, hide_index=True)
                    st.caption("Positive contrib pushes Home (ATS) or Over (Total); negative the opposite.")

            feats = row.get("_features") or {}
            if feats:
                with st.expander("Raw feature values"):
                    st.json(feats)

            st.download_button(
                "Download results CSV",
                data=show.to_csv(index=False).encode("utf-8"),
                file_name=f"byoa_{str(cfg.get('name', 'algo')).replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        elif result is not None:
            st.warning("No rows scored. Enable factors or lower the minimum strength.")
    else:
        st.caption("Load games above, then run the algorithm.")

    st.markdown("---")
    with st.expander("How scoring works"):
        st.markdown(
            """
1. Each **enabled** factor is scaled to a similar range, then multiplied by its **weight**.
2. Contributions are summed into one **BYOA Score**.
3. For **ATS**: positive → Home lean, negative → Away lean.
4. For **Total**: positive → Over, negative → Under.
5. If |score| is below your **minimum strength**, the result is **No strong lean**.

**Invert** flips a factor’s sign (e.g. treat a usual home-favored signal as away-favored).
            """
        )
    st.caption(
        f"BYOA · {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        "Linear weighted factors for transparency. Not betting advice."
    )


