"""
Odds math helpers: American odds, vig removal, edge, implied totals.
"""
from __future__ import annotations

from typing import Optional, Tuple


def implied_team_totals(spread: float, total: float) -> Tuple[float, float]:
    """Return (home_implied, away_implied) from home spread + game total."""
    # Home implied = (total - spread) / 2   when spread is home-centric
    # e.g. home -3, total 45 → home 24, away 21
    home = (total - spread) / 2.0
    away = total - home
    return home, away


def american_to_implied_prob(american_odds) -> Optional[float]:
    try:
        o = float(american_odds)
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def remove_vig_two_way(p_home: float, p_away: float) -> Tuple[float, float]:
    s = p_home + p_away
    if s <= 0:
        return 0.5, 0.5
    return p_home / s, p_away / s


def compute_edge(model_prob: float, market_prob: Optional[float]) -> Optional[float]:
    if market_prob is None:
        return None
    return model_prob - market_prob


def american_profit(units: float, american_odds: float, won: bool) -> float:
    if not won:
        return -units
    if american_odds > 0:
        return units * (american_odds / 100.0)
    return units * (100.0 / abs(american_odds))


def clv_spread(line_taken: float, closing_line: float, side: str) -> float:
    """
    Closing Line Value for a spread bet.
    Positive = you got a better number than the close.
    side: "Home" or "Away" from the perspective of the line_taken.
    """
    try:
        taken = float(line_taken)
        close = float(closing_line)
    except (TypeError, ValueError):
        return 0.0
    # If you took Home -3 and it closed -4, you got +1 CLV
    # If you took Away +3 and it closed +2, you got +1 CLV
    side = (side or "").lower()
    if side in ("home", "h"):
        return close - taken  # more negative close is better for home bettor
    if side in ("away", "a"):
        return taken - close
    return 0.0


def clv_total(line_taken: float, closing_total: float, side: str) -> float:
    """
    CLV for totals. Positive means better number than close.
    side: "Over" or "Under"
    """
    try:
        taken = float(line_taken)
        close = float(closing_total)
    except (TypeError, ValueError):
        return 0.0
    side = (side or "").lower()
    if side in ("over", "o"):
        return close - taken  # lower total at close is better for Over
    if side in ("under", "u"):
        return taken - close
    return 0.0
