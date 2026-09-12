"""
Local file persistence for API key, signal history, and bet log.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Paths relative to project root (parent of this package)
_ROOT = Path(__file__).resolve().parent.parent
_API_KEY_FILE = _ROOT / ".odds_api_key"
_SIGNAL_HISTORY_FILE = _ROOT / "signal_history.csv"
_BET_LOG_FILE = _ROOT / "bet_log.csv"
_LINE_OPENS_FILE = _ROOT / "line_opens.json"
_BOARD_LOCKS_FILE = _ROOT / "board_locks.json"


def _load_saved_api_key() -> str:
    try:
        if _API_KEY_FILE.exists():
            return _API_KEY_FILE.read_text().strip()
    except Exception:
        pass
    return st.session_state.get("odds_api_key", "") or ""


def _save_api_key(key: str) -> None:
    try:
        if key:
            _API_KEY_FILE.write_text(key.strip())
            st.session_state["odds_api_key"] = key.strip()
        else:
            if _API_KEY_FILE.exists():
                _API_KEY_FILE.unlink()
            st.session_state.pop("odds_api_key", None)
    except Exception:
        pass


def _load_signal_history() -> pd.DataFrame:
    cols = [
        "id", "week", "game", "kickoff", "recommendation", "confidence",
        "score", "spread", "total", "result", "logged_at", "graded_at",
    ]
    try:
        if _SIGNAL_HISTORY_FILE.exists():
            df = pd.read_csv(_SIGNAL_HISTORY_FILE)
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=cols)


def _save_signal_history(df: pd.DataFrame) -> None:
    try:
        df.to_csv(_SIGNAL_HISTORY_FILE, index=False)
    except Exception:
        pass


def _load_bet_log() -> pd.DataFrame:
    cols = [
        "id", "date", "game", "market", "side", "line", "odds", "units",
        "result", "profit", "clv", "notes", "status",
    ]
    try:
        if _BET_LOG_FILE.exists():
            df = pd.read_csv(_BET_LOG_FILE)
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=cols)


def _save_bet_log(df: pd.DataFrame) -> None:
    try:
        df.to_csv(_BET_LOG_FILE, index=False)
    except Exception:
        pass
