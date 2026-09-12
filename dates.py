"""
Kickoff formatting and NFL week estimation helpers.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd


def format_kickoff(commence_raw: str) -> str:
    """
    Convert Odds API commence_time (UTC ISO) to US/Eastern for display.
    Avoids evening games rolling to the next calendar day in UTC.
    """
    if not commence_raw:
        return ""
    try:
        ts = pd.to_datetime(commence_raw, utc=True)
        try:
            from zoneinfo import ZoneInfo
            ts_et = ts.tz_convert(ZoneInfo("America/New_York"))
        except Exception:
            ts_et = ts.tz_convert(None) - pd.Timedelta(hours=4)
            return ts_et.strftime("%Y-%m-%d %H:%M ET")
        return ts_et.strftime("%Y-%m-%d %H:%M ET")
    except Exception:
        return (commence_raw[:16].replace("T", " ") if len(commence_raw) >= 16 else commence_raw)


def format_schedule_kickoff(gameday: str, gametime: Optional[str]) -> str:
    """Build display kickoff from schedule gameday + gametime (local ET style)."""
    if not gameday:
        return ""
    gd = str(gameday)[:10]
    gt = (str(gametime).strip() if gametime and str(gametime) not in ("None", "nan") else "")
    if gt:
        try:
            hh, mm = gt.split(":")[:2]
            return f"{gd} {int(hh):02d}:{mm} ET"
        except Exception:
            return f"{gd} {gt} ET"
    return f"{gd} ET"


def estimate_week_from_date(game_date: str) -> Optional[int]:
    """
    Estimate NFL week from calendar date when schedule lookup fails.
    NFL weeks run roughly Thursday → following Wednesday (MNF included).
    This is a best-effort approximation; prefer schedule-based week when available.
    """
    try:
        d = pd.to_datetime(str(game_date)[:10]).date()
    except Exception:
        return None

    # Approximate season start (first Thursday after Labor Day varies; use early September)
    year = d.year
    # Labor Day is first Monday in September
    labor = datetime(year, 9, 1).date()
    while labor.weekday() != 0:  # Monday
        labor = labor.replace(day=labor.day + 1)
    # Week 1 typically starts the Thursday after Labor Day
    season_start = labor.replace(day=labor.day + 3)  # Thursday

    if d < season_start:
        # Could be previous season late games / playoffs — rough
        return None

    delta = (d - season_start).days
    week = delta // 7 + 1
    if week < 1:
        return 1
    if week > 22:  # regular + playoffs buffer
        return 22
    return week


def current_nfl_week() -> Optional[int]:
    """Best-effort current NFL week from today's date."""
    try:
        return estimate_week_from_date(datetime.now().strftime("%Y-%m-%d"))
    except Exception:
        return None
