"""
Team identity, division, timezone, and travel helpers.
"""
from __future__ import annotations

from typing import Optional, Set

from config.constants import (
    TEAM_NAME_TO_ABBR,
    ABBR_TO_FULL,
    TEAM_ALIASES,
    TEAM_TO_DIV,
    TEAM_TZ,
)


def to_abbr(name: str) -> Optional[str]:
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if name in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[name]
    if len(name) <= 3 and name.isupper():
        return name
    return None


def full_name(abbr: str) -> str:
    return ABBR_TO_FULL.get(abbr, abbr)


def expand_team(t: str) -> Set[str]:
    """Return set of known aliases for a team abbreviation."""
    return TEAM_ALIASES.get(t, {t}) | {t}


def is_divisional(home: str, away: str) -> bool:
    return TEAM_TO_DIV.get(home) == TEAM_TO_DIV.get(away) and home in TEAM_TO_DIV


def timezone_diff(home: str, away: str) -> int:
    """Absolute hours of timezone change for the away team traveling to home."""
    h = TEAM_TZ.get(home, -5)
    a = TEAM_TZ.get(away, -5)
    return abs(h - a)


def travel_direction(home: str, away: str) -> str:
    """Rough direction of travel for away team: Eastbound, Westbound, or None."""
    h = TEAM_TZ.get(home, -5)
    a = TEAM_TZ.get(away, -5)
    diff = h - a  # positive = away is traveling west (to earlier TZ)
    if abs(diff) < 1:
        return "None"
    return "Westbound" if diff > 0 else "Eastbound"


def normalize_team_abbr(t: str) -> str:
    """Normalize various team codes to the canonical 2–3 letter abbr used in the app."""
    if not t:
        return t
    t = str(t).strip().upper()
    # Common nflverse / ESPN variants
    mapping = {
        "LAR": "LA", "STL": "LA",
        "SD": "LAC",
        "OAK": "LV", "LVR": "LV",
        "WSH": "WAS", "WFT": "WAS",
        "GNB": "GB",
        "KAN": "KC",
        "NWE": "NE",
        "NOR": "NO",
        "SFO": "SF",
        "TAM": "TB",
        "JAC": "JAX",
    }
    return mapping.get(t, t)
