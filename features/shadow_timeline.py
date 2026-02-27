"""Shadow Workflow Timeline -- Gantt-style visualization of formal vs shadow activities.

Renders a horizontal bar chart showing when each activity occurred during the
meeting, color-coded by formal (green) or shadow/deviation (red).
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from pipeline.time_utils import ts_to_seconds


# ── public API ───────────────────────────────────────────────────────


def build_timeline_data(
    mapped_events: pd.DataFrame,
    agenda_activities: list,
) -> dict:
    """Group events by mapped_activity and compute start/end times.

    Parameters
    ----------
    mapped_events : pd.DataFrame
        Must contain ``timestamp``, ``activity_name``, ``mapped_activity``,
        and ``source`` columns.
    agenda_activities : list
        Ordered agenda labels used to distinguish formal from shadow items.

    Returns
    -------
    dict
        ``formal_segments``, ``shadow_segments`` (each a list of dicts with
        *label*, *start_sec*, *end_sec*), and ``total_duration_sec``.
    """
    empty = {"formal_segments": [], "shadow_segments": [], "total_duration_sec": 0}

    if mapped_events.empty or "mapped_activity" not in mapped_events.columns:
        return empty

    df = mapped_events.copy()
    df["_sec"] = df["timestamp"].apply(ts_to_seconds)

    agenda_set = {a.strip() for a in agenda_activities}

    formal_segments: list[dict] = []
    shadow_segments: list[dict] = []

    for label, grp in df.groupby("mapped_activity", sort=False):
        label = str(label)
        start_sec = int(grp["_sec"].min())
        end_sec = int(grp["_sec"].max())
        segment = {"label": label, "start_sec": start_sec, "end_sec": end_sec}

        if label.startswith("Deviation:"):
            shadow_segments.append(segment)
        else:
            formal_segments.append(segment)

    total_duration = int(df["_sec"].max()) if not df.empty else 0

    return {
        "formal_segments": sorted(formal_segments, key=lambda s: s["start_sec"]),
        "shadow_segments": sorted(shadow_segments, key=lambda s: s["start_sec"]),
        "total_duration_sec": total_duration,
    }


def render_timeline(timeline_data: dict) -> None:
    """Render a Gantt-style timeline chart in Streamlit with summary stats.

    Parameters
    ----------
    timeline_data : dict
        Output of :func:`build_timeline_data`.
    """
    formal = timeline_data.get("formal_segments", [])
    shadow = timeline_data.get("shadow_segments", [])
    total_dur = timeline_data.get("total_duration_sec", 0)

    if not formal and not shadow:
        st.info("No timeline data to display.")
        return

    # -- build a single DataFrame for Altair --------------------------------
    rows = []
    for seg in formal:
        rows.append({
            "Activity": seg["label"],
            "Start (min)": seg["start_sec"] / 60,
            "End (min)": max(seg["end_sec"], seg["start_sec"] + 1) / 60,
            "Type": "Formal",
        })
    for seg in shadow:
        rows.append({
            "Activity": seg["label"],
            "Start (min)": seg["start_sec"] / 60,
            "End (min)": max(seg["end_sec"], seg["start_sec"] + 1) / 60,
            "Type": "Shadow",
        })

    chart_df = pd.DataFrame(rows)

    # -- Altair Gantt chart --------------------------------------------------
    color_scale = alt.Scale(
        domain=["Formal", "Shadow"],
        range=["#059669", "#dc2626"],
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Start (min):Q", title="Time (minutes)"),
            x2="End (min):Q",
            y=alt.Y("Activity:N", sort=None, title=None),
            color=alt.Color("Type:N", scale=color_scale, legend=alt.Legend(title="Type")),
            tooltip=["Activity", "Type", "Start (min)", "End (min)"],
        )
        .properties(height=max(len(rows) * 28, 120))
    )

    st.subheader("Shadow Workflow Timeline")
    st.altair_chart(chart, use_container_width=True)

    # -- summary stats -------------------------------------------------------
    formal_time = sum(max(s["end_sec"] - s["start_sec"], 1) for s in formal)
    shadow_time = sum(max(s["end_sec"] - s["start_sec"], 1) for s in shadow)
    combined = formal_time + shadow_time

    formal_pct = (formal_time / combined * 100) if combined else 0
    shadow_pct = (shadow_time / combined * 100) if combined else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Formal time", f"{formal_time // 60}m {formal_time % 60}s", f"{formal_pct:.1f}%")
    c2.metric("Shadow time", f"{shadow_time // 60}m {shadow_time % 60}s", f"{shadow_pct:.1f}%")
    c3.metric("Meeting duration", f"{total_dur // 60}m {total_dur % 60}s")
