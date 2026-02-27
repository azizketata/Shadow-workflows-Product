"""Full Transcript View -- color-coded, scrollable transcript of meeting events.

Each line is categorised as *formal* (mapped to an agenda item), *shadow*
(mapped but flagged as a deviation), or *noise* (unmapped), and rendered
with a matching background colour in Streamlit.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline.time_utils import ts_to_seconds


# ── colour constants ─────────────────────────────────────────────────
_COLORS = {
    "formal": "#e6ffe6",
    "shadow": "#fde8e8",
    "noise":  "#f0f0f0",
}


# ── public API ───────────────────────────────────────────────────────

def build_annotated_transcript(
    raw_events: pd.DataFrame,
    mapped_events: pd.DataFrame = None,
) -> list[dict]:
    """Turn raw video events into an ordered, categorised transcript.

    Parameters
    ----------
    raw_events : pd.DataFrame
        The original video events with ``original_text`` column.
    mapped_events : pd.DataFrame, optional
        If provided, used to annotate each raw event with its
        ``mapped_activity`` by closest timestamp matching.

    Returns
    -------
    list[dict]
        Each dict contains: timestamp, seconds, text, activity,
        mapped_to, category (formal | shadow | noise), source.
    """
    if raw_events is None or raw_events.empty:
        return []

    df = raw_events.copy()

    # Determine which column has transcript text
    text_col = None
    for col in ("original_text", "details", "activity_name"):
        if col in df.columns:
            non_empty = df[col].fillna("").str.strip().astype(bool)
            if non_empty.any():
                text_col = col
                break

    if text_col is None:
        return []

    # Keep only rows with real text content
    df = df[df[text_col].fillna("").str.strip().astype(bool)].copy()
    if df.empty:
        return []

    # Build a timestamp → mapped_activity lookup from mapped_events
    mapped_lookup = {}
    if mapped_events is not None and not mapped_events.empty and "mapped_activity" in mapped_events.columns:
        for _, row in mapped_events.iterrows():
            sec = ts_to_seconds(row.get("timestamp", 0))
            mapped_lookup[sec] = str(row["mapped_activity"])

    transcript: list[dict] = []
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text:
            continue

        sec = ts_to_seconds(row.get("timestamp", 0))

        # Try to find a mapped activity for this timestamp (exact or nearest)
        mapped = mapped_lookup.get(sec, "")
        if not mapped and mapped_lookup:
            # Find nearest mapped timestamp within 30 seconds
            closest_sec = min(mapped_lookup.keys(), key=lambda s: abs(s - sec))
            if abs(closest_sec - sec) <= 30:
                mapped = mapped_lookup[closest_sec]

        if not mapped:
            category = "noise"
        elif mapped.startswith("Deviation:"):
            category = "shadow"
        else:
            category = "formal"

        transcript.append({
            "timestamp": str(row.get("timestamp", "00:00:00")),
            "seconds":   sec,
            "text":      text,
            "activity":  str(row.get("activity_name", "")),
            "mapped_to": mapped,
            "category":  category,
            "source":    str(row.get("source", "")),
        })

    transcript.sort(key=lambda e: e["seconds"])
    return transcript


def render_transcript(transcript: list) -> None:
    """Render a scrollable, colour-coded transcript in Streamlit.

    Parameters
    ----------
    transcript : list[dict]
        Output of :func:`build_annotated_transcript`.
    """
    if not transcript:
        st.info("No transcript entries to display.")
        return

    st.subheader("Full Transcript")

    lines: list[str] = []
    for entry in transcript:
        bg = _COLORS.get(entry["category"], _COLORS["noise"])
        mapped_label = f' [{entry["mapped_to"]}]' if entry["mapped_to"] else ""
        lines.append(
            f"<div style='background:{bg};padding:6px 10px;"
            f"border-radius:4px;margin-bottom:4px;font-size:0.92em'>"
            f"<strong>{entry['timestamp']}</strong>"
            f"<span style='color:#555'>{mapped_label}</span> "
            f"{entry['text']}"
            f"</div>"
        )

    html = (
        "<div style='max-height:520px;overflow-y:auto;"
        "border:1px solid #ddd;border-radius:6px;padding:6px'>"
        + "\n".join(lines)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)
