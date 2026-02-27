"""Agenda Item Duration Analysis -- time-span per mapped activity.

Computes how long each agenda item occupied in the meeting, flags
anomalies (unusually long or short items), and renders an Altair
bar chart with a summary table in Streamlit.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt

from pipeline.time_utils import ts_to_seconds, seconds_to_ts


# ── public API ───────────────────────────────────────────────────────


def compute_durations(
    mapped_events: pd.DataFrame,
    agenda_activities: list,
) -> list[dict]:
    """Compute the time span each agenda item occupies in the meeting.

    Parameters
    ----------
    mapped_events : pd.DataFrame
        Must contain ``timestamp`` and ``mapped_activity`` columns.
    agenda_activities : list
        Ordered agenda item labels.

    Returns
    -------
    list[dict]
        Each dict has keys: item, duration_min, start, end,
        pct_of_meeting, anomaly.  Sorted by start time.
    """
    if mapped_events.empty or "mapped_activity" not in mapped_events.columns:
        return []

    df = mapped_events.copy()
    df["_sec"] = df["timestamp"].apply(ts_to_seconds)

    # Total meeting span (first event to last event)
    meeting_start = df["_sec"].min()
    meeting_end = df["_sec"].max()
    total_span = max(meeting_end - meeting_start, 1)  # avoid div-by-zero

    # Group by mapped_activity
    groups = df.groupby("mapped_activity")["_sec"]
    records: list[dict] = []

    for activity, secs in groups:
        start_sec = int(secs.min())
        end_sec = int(secs.max())
        span = end_sec - start_sec
        duration_min = round(span / 60.0, 2)
        pct = round((span / total_span) * 100, 1)

        records.append({
            "item": str(activity),
            "duration_min": duration_min,
            "start": seconds_to_ts(start_sec),
            "end": seconds_to_ts(end_sec),
            "start_sec": start_sec,
            "pct_of_meeting": pct,
            "anomaly": None,
        })

    if not records:
        return []

    # Sort by start time
    records.sort(key=lambda r: r["start_sec"])

    # Anomaly detection based on average duration
    durations = [r["duration_min"] for r in records]
    avg_dur = sum(durations) / len(durations) if durations else 0

    if avg_dur > 0:
        for r in records:
            if r["duration_min"] > 2.0 * avg_dur:
                r["anomaly"] = "longest"
            elif r["duration_min"] < 0.25 * avg_dur:
                r["anomaly"] = "shortest"

    # Drop internal helper key
    for r in records:
        r.pop("start_sec", None)

    return records


# ── Streamlit rendering ──────────────────────────────────────────────


def render_duration_analysis(durations: list) -> None:
    """Render an Altair bar chart and summary table in Streamlit.

    Parameters
    ----------
    durations : list[dict]
        Output of :func:`compute_durations`.
    """
    if not durations:
        st.info("No duration data to display.")
        return

    st.subheader("Agenda Item Duration Analysis")

    chart_df = pd.DataFrame(durations)

    # Assign color based on anomaly flag
    def _bar_color(anomaly):
        if anomaly == "longest":
            return "amber"
        if anomaly == "shortest":
            return "gray"
        return "steelblue"

    chart_df["color"] = chart_df["anomaly"].apply(_bar_color)

    # Altair horizontal bar chart
    color_scale = alt.Scale(
        domain=["steelblue", "amber", "gray"],
        range=["steelblue", "#FFBF00", "#999999"],
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y("item:N", sort=None, title="Agenda Item"),
            x=alt.X("duration_min:Q", title="Duration (minutes)"),
            color=alt.Color(
                "color:N",
                scale=color_scale,
                legend=alt.Legend(title="Flag"),
            ),
            tooltip=[
                alt.Tooltip("item:N", title="Item"),
                alt.Tooltip("duration_min:Q", title="Duration (min)"),
                alt.Tooltip("pct_of_meeting:Q", title="% of Meeting"),
                alt.Tooltip("start:N", title="Start"),
                alt.Tooltip("end:N", title="End"),
            ],
        )
        .properties(height=max(len(durations) * 28, 150))
    )

    st.altair_chart(chart, use_container_width=True)

    # Summary table
    table_df = pd.DataFrame(
        [
            {
                "Item": d["item"],
                "Duration": f"{d['duration_min']} min",
                "% of Meeting": f"{d['pct_of_meeting']}%",
                "Flag": d["anomaly"] or "",
            }
            for d in durations
        ]
    )
    st.dataframe(table_df, use_container_width=True, hide_index=True)
