"""Citizen Alert System — subscribe to topics and get notified when they appear
in meeting transcripts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

ALERTS_DB = "citizen_alerts.db"


def _db_path() -> Path:
    return Path(ALERTS_DB)


def _init_alerts_db() -> None:
    """Create the subscriptions table if it does not exist."""
    conn = sqlite3.connect(_db_path())
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def add_subscription(topic: str, email: str = None) -> int:
    """Insert a new subscription and return its id."""
    _init_alerts_db()
    conn = sqlite3.connect(_db_path())
    try:
        cur = conn.execute(
            "INSERT INTO subscriptions (topic, email) VALUES (?, ?)",
            (topic, email),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_subscriptions() -> list[dict]:
    """Return all subscriptions as a list of dicts."""
    _init_alerts_db()
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, topic, email, created_at FROM subscriptions ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _delete_subscription(sub_id: int) -> None:
    """Remove a subscription by id."""
    conn = sqlite3.connect(_db_path())
    try:
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (sub_id,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Alert matching
# ---------------------------------------------------------------------------

def check_alerts(
    mapped_events: pd.DataFrame,
    subscriptions: list[dict],
) -> list[dict]:
    """Check mapped events against subscription topics.

    Parameters
    ----------
    mapped_events : DataFrame with columns timestamp, activity_name,
                    original_text, mapped_activity.
    subscriptions : list of subscription dicts (must contain 'topic' key).

    Returns
    -------
    list of dicts: {"topic": str, "matches": [{"timestamp", "text", "activity"}]}
    """
    alerts: list[dict] = []
    for sub in subscriptions:
        keyword = sub["topic"].lower()
        matches: list[dict] = []
        for _, row in mapped_events.iterrows():
            orig = str(row.get("original_text", "")).lower()
            act = str(row.get("activity_name", "")).lower()
            if keyword in orig or keyword in act:
                matches.append({
                    "timestamp": str(row.get("timestamp", "")),
                    "text": str(row.get("original_text", "")),
                    "activity": str(row.get("activity_name", "")),
                })
        if matches:
            alerts.append({"topic": sub["topic"], "matches": matches})
    return alerts


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render_alerts(mapped_events: pd.DataFrame = None) -> None:
    """Render the Citizen Alert System panel in Streamlit."""
    st.subheader("Citizen Alert Subscriptions")

    # -- Add new subscription ------------------------------------------------
    col_topic, col_btn = st.columns([4, 1])
    with col_topic:
        new_topic = st.text_input("Subscribe to a topic", key="alert_new_topic")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)  # vertical alignment
        add_clicked = st.button("Subscribe", key="alert_add_btn")

    if add_clicked and new_topic.strip():
        add_subscription(new_topic.strip())
        st.success(f"Subscribed to **{new_topic.strip()}**")
        st.rerun()

    # -- Current subscriptions -----------------------------------------------
    subs = get_subscriptions()
    if not subs:
        st.info("No subscriptions yet. Enter a topic above to get started.")
        return

    st.markdown(f"**{len(subs)}** active subscription(s)")
    for sub in subs:
        col_label, col_del = st.columns([5, 1])
        with col_label:
            email_note = f" ({sub['email']})" if sub.get("email") else ""
            st.markdown(f"- **{sub['topic']}**{email_note}")
        with col_del:
            if st.button("Delete", key=f"alert_del_{sub['id']}"):
                _delete_subscription(sub["id"])
                st.rerun()

    # -- Matched alerts ------------------------------------------------------
    if mapped_events is not None and not mapped_events.empty:
        alerts = check_alerts(mapped_events, subs)
        if alerts:
            st.subheader("Triggered Alerts")
            for alert in alerts:
                with st.expander(
                    f"{alert['topic']} — {len(alert['matches'])} match(es)"
                ):
                    for m in alert["matches"]:
                        st.markdown(
                            f"**{m['timestamp']}** | {m['activity']}  \n"
                            f"> {m['text']}"
                        )
        else:
            st.info("No alerts triggered for the current transcript.")
