"""Multi-Meeting Trend Dashboard -- SQLite-backed history and cross-session analytics.

Persists meeting results in a local SQLite database so users can track
fitness, agenda coverage, and shadow workflow frequency over time.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

DB_PATH = "meeting_history.db"

# ── schema helpers ───────────────────────────────────────────────────


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the meeting-history database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    """Create tables if they do not already exist."""
    conn = _get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, date TEXT,
                fitness_dedup REAL, fitness_raw REAL,
                agenda_coverage REAL, shadow_count INTEGER,
                matched_count INTEGER, event_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS meeting_shadows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER, shadow_label TEXT, count INTEGER,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id)
            );
        """)
        conn.commit()
    finally:
        conn.close()


# ── data access ──────────────────────────────────────────────────────


def save_meeting_result(
    name: str,
    date: str,
    fitness_dedup: float,
    fitness_raw: float,
    agenda_coverage: float,
    shadow_count: int,
    matched_count: int,
    event_count: int,
    shadow_labels: list,
) -> int:
    """Insert a meeting result and its shadow labels.  Returns the new meeting id."""
    _init_db()
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO meetings (name,date,fitness_dedup,fitness_raw,"
            "agenda_coverage,shadow_count,matched_count,event_count) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (name, date, fitness_dedup, fitness_raw,
             agenda_coverage, shadow_count, matched_count, event_count),
        )
        meeting_id = cursor.lastrowid

        # Aggregate duplicate shadow labels into (label, count) pairs
        label_counts = Counter(shadow_labels)
        for label, cnt in label_counts.items():
            conn.execute(
                "INSERT INTO meeting_shadows (meeting_id, shadow_label, count) VALUES (?, ?, ?)",
                (meeting_id, label, cnt),
            )

        conn.commit()
        return meeting_id
    finally:
        conn.close()


def load_meeting_history() -> pd.DataFrame:
    """Load all meetings ordered by date.  Returns an empty DataFrame if none exist."""
    _init_db()
    conn = _get_connection()
    try:
        return pd.read_sql_query(
            "SELECT * FROM meetings ORDER BY date ASC, id ASC", conn)
    finally:
        conn.close()


def _load_shadow_frequencies() -> pd.DataFrame:
    """Aggregate shadow labels across all meetings, sorted by frequency."""
    _init_db()
    conn = _get_connection()
    try:
        return pd.read_sql_query(
            "SELECT shadow_label, SUM(count) AS total_count, "
            "COUNT(DISTINCT meeting_id) AS meeting_appearances "
            "FROM meeting_shadows GROUP BY shadow_label ORDER BY total_count DESC",
            conn)
    finally:
        conn.close()


# ── Streamlit UI ─────────────────────────────────────────────────────


def render_trend_dashboard() -> None:
    """Render the multi-meeting trend dashboard inside Streamlit."""

    st.header("Multi-Meeting Trend Dashboard")

    # ── save form ────────────────────────────────────────────────────
    with st.expander("Save Meeting Results", expanded=False):
        col_name, col_date = st.columns(2)
        meeting_name = col_name.text_input(
            "Meeting name", value="", key="trend_meeting_name",
        )
        meeting_date = col_date.text_input(
            "Date (YYYY-MM-DD)", value=str(date.today()), key="trend_meeting_date",
        )

        c1, c2 = st.columns(2)
        fitness_dedup = c1.number_input("Fitness (dedup) %", 0.0, 100.0, 0.0, key="trend_fit_d")
        fitness_raw = c2.number_input("Fitness (raw) %", 0.0, 100.0, 0.0, key="trend_fit_r")

        c3, c4 = st.columns(2)
        agenda_cov = c3.number_input("Agenda coverage %", 0.0, 100.0, 0.0, key="trend_ag_cov")
        shadow_cnt = c4.number_input("Shadow count", 0, 10000, 0, key="trend_sh_cnt")

        c5, c6 = st.columns(2)
        matched_cnt = c5.number_input("Matched events", 0, 100000, 0, key="trend_match")
        event_cnt = c6.number_input("Total events", 0, 100000, 0, key="trend_evt")

        shadow_labels_raw = st.text_input(
            "Shadow labels (comma-separated)",
            value="",
            key="trend_shadow_labels",
        )

        if st.button("Save to history", key="trend_save_btn"):
            if not meeting_name.strip():
                st.warning("Please enter a meeting name.")
            else:
                labels = [
                    s.strip() for s in shadow_labels_raw.split(",") if s.strip()
                ]
                mid = save_meeting_result(
                    name=meeting_name.strip(),
                    date=meeting_date.strip(),
                    fitness_dedup=fitness_dedup,
                    fitness_raw=fitness_raw,
                    agenda_coverage=agenda_cov,
                    shadow_count=int(shadow_cnt),
                    matched_count=int(matched_cnt),
                    event_count=int(event_cnt),
                    shadow_labels=labels,
                )
                st.success(f"Saved meeting **{meeting_name}** (id={mid}).")

    # ── load history ─────────────────────────────────────────────────
    history = load_meeting_history()

    if history.empty:
        st.info("No meeting history yet. Save a result above to get started.")
        return

    # ── fitness trend chart ──────────────────────────────────────────
    st.subheader("Fitness Over Time")

    chart_df = history[["date", "name", "fitness_dedup", "fitness_raw"]].copy()
    chart_df = chart_df.melt(
        id_vars=["date", "name"],
        value_vars=["fitness_dedup", "fitness_raw"],
        var_name="metric",
        value_name="score",
    )
    chart_df["metric"] = chart_df["metric"].map({
        "fitness_dedup": "Dedup Fitness",
        "fitness_raw": "Raw Fitness",
    })

    line = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Meeting Date"),
            y=alt.Y("score:Q", title="Fitness %", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["name:N", "date:T", "metric:N", "score:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(line, use_container_width=True)

    # ── meeting history table ────────────────────────────────────────
    st.subheader("Meeting History")

    display_cols = [
        "id", "name", "date", "fitness_dedup", "fitness_raw",
        "agenda_coverage", "shadow_count", "matched_count", "event_count",
    ]
    available = [c for c in display_cols if c in history.columns]
    st.dataframe(history[available], use_container_width=True, hide_index=True)

    # ── shadow frequency table ───────────────────────────────────────
    st.subheader("Shadow Workflow Frequency")

    shadow_df = _load_shadow_frequencies()
    if shadow_df.empty:
        st.info("No shadow labels recorded yet.")
    else:
        st.dataframe(shadow_df, use_container_width=True, hide_index=True)
