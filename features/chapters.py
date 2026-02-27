"""Meeting Chaptering -- groups mapped events into sequential chapters.

Each chapter represents a contiguous block of events sharing the same
mapped_activity.  Chapters whose label starts with ``Deviation:`` are
flagged as shadow workflows.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline.time_utils import ts_to_seconds, seconds_to_ts


# ── public API ───────────────────────────────────────────────────────

def build_chapters(
    mapped_events: pd.DataFrame,
    agenda_activities: list,
) -> list[dict]:
    """Group consecutive events by *mapped_activity* into chapters.

    Parameters
    ----------
    mapped_events : pd.DataFrame
        Must contain at least ``timestamp`` and ``mapped_activity`` columns.
    agenda_activities : list
        Ordered agenda labels (used only for reference; shadow detection
        relies on the ``Deviation:`` prefix).

    Returns
    -------
    list[dict]
        Each dict: label, start, end, start_sec, end_sec, event_count,
        is_shadow.
    """
    if mapped_events.empty or "mapped_activity" not in mapped_events.columns:
        return []

    df = mapped_events.copy()
    df["_sec"] = df["timestamp"].apply(ts_to_seconds)
    df = df.sort_values("_sec").reset_index(drop=True)

    chapters: list[dict] = []
    current_label: str | None = None
    start_sec = 0
    end_sec = 0
    count = 0

    for _, row in df.iterrows():
        label = str(row.get("mapped_activity", "Unknown"))
        sec = int(row["_sec"])

        if label != current_label:
            # flush previous chapter
            if current_label is not None:
                chapters.append(_make_chapter(current_label, start_sec, end_sec, count))
            current_label = label
            start_sec = sec
            end_sec = sec
            count = 1
        else:
            end_sec = sec
            count += 1

    # flush last chapter
    if current_label is not None:
        chapters.append(_make_chapter(current_label, start_sec, end_sec, count))

    return chapters


def render_chapters(chapters: list[dict], uploaded_video) -> None:
    """Render a chapter timeline in Streamlit with video seek support.

    Parameters
    ----------
    chapters : list[dict]
        Output of :func:`build_chapters`.
    uploaded_video
        A Streamlit ``UploadedFile`` (or file-path) for ``st.video``.
    """
    if not chapters:
        st.info("No chapters to display.")
        return

    st.subheader("Meeting Chapters")

    for idx, ch in enumerate(chapters):
        is_shadow = ch["is_shadow"]
        bg = "#fdd" if is_shadow else "#dfd"
        tag = "Shadow" if is_shadow else "Formal"
        label = ch["label"]
        time_range = f"{ch['start']} -- {ch['end']}"

        st.markdown(
            f"<div style='background:{bg};padding:8px 12px;border-radius:6px;"
            f"margin-bottom:6px'>"
            f"<strong>{idx + 1}. {label}</strong> "
            f"<span style='float:right'>{time_range} &middot; "
            f"{ch['event_count']} events &middot; "
            f"<em>{tag}</em></span></div>",
            unsafe_allow_html=True,
        )

        with st.expander(f"Play from {ch['start']}"):
            st.video(uploaded_video, start_time=ch["start_sec"])


# ── internal helpers ─────────────────────────────────────────────────

def _make_chapter(label: str, start_sec: int, end_sec: int, count: int) -> dict:
    return {
        "label": label,
        "start": seconds_to_ts(start_sec),
        "end": seconds_to_ts(end_sec),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "event_count": count,
        "is_shadow": label.startswith("Deviation:"),
    }
