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

def build_annotated_transcript(mapped_events: pd.DataFrame) -> list[dict]:
    """Turn mapped events into an ordered, categorised transcript.

    Parameters
    ----------
    mapped_events : pd.DataFrame
        Expected columns: ``timestamp``, ``activity_name``, ``source``,
        ``original_text``, ``mapped_activity``.

    Returns
    -------
    list[dict]
        Each dict contains: timestamp, seconds, text, activity,
        mapped_to, category (formal | shadow | noise), source.
    """
    if mapped_events.empty:
        return []

    df = mapped_events.copy()

    # Keep only rows with real transcript text
    if "original_text" not in df.columns:
        return []
    df = df[df["original_text"].fillna("").str.strip().astype(bool)].copy()
    if df.empty:
        return []

    transcript: list[dict] = []
    for _, row in df.iterrows():
        mapped = str(row.get("mapped_activity", "")) if pd.notna(row.get("mapped_activity")) else ""
        if not mapped:
            category = "noise"
        elif mapped.startswith("Deviation:"):
            category = "shadow"
        else:
            category = "formal"

        transcript.append({
            "timestamp": str(row.get("timestamp", "00:00:00")),
            "seconds":   ts_to_seconds(row.get("timestamp", 0)),
            "text":      str(row["original_text"]).strip(),
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
