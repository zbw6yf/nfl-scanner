"""
Theme injection, logo helpers, and small UI components.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import streamlit as st

from config.constants import CONF_COLORS


def inject_theme_css(theme: str) -> None:
    dark = theme == "Dark"
    bg = "#0e1117" if dark else "#f7f8fa"
    card = "#1a1f2e" if dark else "#ffffff"
    text = "#e8eaed" if dark else "#1a1d26"
    muted = "#9aa0a6" if dark else "#5f6368"
    accent = "#3b82f6"
    border = "#2d3348" if dark else "#e5e7eb"
    st.markdown(
        f"""
<style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
    h1, h2, h3, h4 {{ letter-spacing: -0.02em; }}
    div[data-testid="stMetric"] {{
        background: {card};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }}
    div[data-testid="stMetric"] label {{ color: {muted} !important; }}
    .nsc-hero {{
        background: linear-gradient(135deg, #0b1220 0%, #1e3a5f 55%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        color: #f8fafc;
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .nsc-hero h1 {{
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc !important;
    }}
    .nsc-hero p {{
        margin: 0.35rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.95rem;
    }}
    .nsc-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        background: rgba(255,255,255,0.12);
        color: #e2e8f0;
    }}
    .nsc-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.65rem;
    }}
    .nsc-card-title {{ font-weight: 650; font-size: 1.02rem; margin-bottom: 0.25rem; color: {text}; }}
    .nsc-muted {{ color: {muted}; font-size: 0.85rem; }}
    .nsc-pill {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 4px;
    }}
    .nsc-stamp {{
        color: {muted};
        font-size: 0.8rem;
        margin: 0.15rem 0 0.75rem 0;
    }}
    .nsc-footer {{
        margin-top: 2rem;
        padding-top: 0.75rem;
        border-top: 1px solid {border};
        color: {muted};
        font-size: 0.8rem;
    }}
    [data-testid="stSidebar"] {{
        background: {"#111827" if dark else "#ffffff"};
    }}
</style>
        """,
        unsafe_allow_html=True,
    )


def get_logo_data_uri() -> Optional[str]:
    for p in [
        Path(__file__).resolve().parent.parent / "tailme_logo.png",
        Path("/home/workdir/artifacts/tailme_logo.png"),
        Path("tailme_logo.png"),
    ]:
        if p.exists():
            try:
                data = p.read_bytes()
                b64 = base64.b64encode(data).decode("ascii")
                return f"data:image/png;base64,{b64}"
            except Exception:
                continue
    return None


def conf_pill(grade: str) -> str:
    g = (grade or "F").upper()[:1]
    color = CONF_COLORS.get(g, "#6b7280")
    return (
        f'<span class="nsc-pill" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55">{g}</span>'
    )


def stamp_now(key: str) -> None:
    from datetime import datetime
    st.session_state[f"updated_{key}"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp_text(key: str, label: str) -> str:
    val = st.session_state.get(f"updated_{key}")
    if not val:
        return f"{label}: —"
    return f"{label}: {val}"


def last_update_caption(*keys: str, label: str = "Last update") -> str:
    times = []
    for key in keys:
        val = st.session_state.get(f"updated_{key}")
        if val:
            times.append(str(val))
    if not times:
        return f"{label}: not yet refreshed this session"
    times.sort()
    return f"{label}: {times[-1]}"
