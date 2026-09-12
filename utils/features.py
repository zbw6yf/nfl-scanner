"""
Feature flags, user tier, and upgrade CTA helpers.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from config.constants import DEFAULT_FEATURE_FLAGS


def _secrets_features() -> dict:
    try:
        feat = st.secrets.get("features", {})
        return dict(feat) if feat else {}
    except Exception:
        return {}


def feature_enabled(name: str) -> bool:
    flags = dict(DEFAULT_FEATURE_FLAGS)
    flags.update(_secrets_features())
    overrides = st.session_state.get("feature_flag_overrides") or {}
    flags.update(overrides)
    return bool(flags.get(name, False))


def user_tier() -> str:
    """free | pro — set via session after Stripe webhook / login (future)."""
    tier = st.session_state.get("user_tier")
    if tier in ("free", "pro"):
        return tier
    try:
        if st.secrets.get("user_tier") in ("free", "pro"):
            return str(st.secrets.get("user_tier"))
    except Exception:
        pass
    return "free"


def require_pro(flag_name: str) -> bool:
    """Return True if the user is blocked (needs Pro)."""
    if not feature_enabled(flag_name):
        return False
    if not feature_enabled("stripe_enabled"):
        return False  # soft launch: flags on but stripe off = allow all
    return user_tier() != "pro"


def stripe_payment_link() -> str:
    try:
        return str(st.secrets.get("stripe", {}).get("payment_link") or "")
    except Exception:
        return ""


def render_upgrade_cta(context: str = "") -> None:
    if not feature_enabled("show_upgrade_cta"):
        return
    if not feature_enabled("stripe_enabled"):
        return
    if user_tier() == "pro":
        return
    link = stripe_payment_link()
    msg = "Pro unlocks full Signal History, props, bankroll tools, and priority board refresh."
    if context:
        msg = f"{context} {msg}"
    st.warning(msg)
    if link:
        st.link_button("Upgrade to Pro", link, type="primary")
    else:
        st.caption("Set secrets.stripe.payment_link to enable checkout.")
