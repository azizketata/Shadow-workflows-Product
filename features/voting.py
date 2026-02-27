"""Voting Record Extraction -- identifies and summarises votes from mapped events.

Combines audio vote motions (keyword matches) with visual hand-raise counts
to produce a consolidated voting record for the meeting.
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from pipeline.time_utils import ts_to_seconds


# ── keyword patterns ─────────────────────────────────────────────────

_VOTE_KEYWORDS = re.compile(
    r"\b(vote|motion|confirmed vote|roll call|ayes?|nays?|abstain|ballot)\b",
    re.IGNORECASE,
)

_OUTCOME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ayes?\s+have\s+it", re.I), "passed"),
    (re.compile(r"motion\s+carries", re.I), "passed"),
    (re.compile(r"\bapproved\b", re.I), "passed"),
    (re.compile(r"\badopted\b", re.I), "passed"),
    (re.compile(r"\bunanimous(ly)?\b", re.I), "passed"),
    (re.compile(r"\bpassed\b", re.I), "passed"),
    (re.compile(r"motion\s+fails", re.I), "failed"),
    (re.compile(r"motion\s+denied", re.I), "failed"),
    (re.compile(r"\brejected\b", re.I), "failed"),
    (re.compile(r"\bdefeated\b", re.I), "failed"),
    (re.compile(r"\bnays?\s+have\s+it", re.I), "failed"),
]


# ── public API ───────────────────────────────────────────────────────

def extract_voting_records(mapped_events: pd.DataFrame) -> list[dict]:
    """Extract voting records from mapped meeting events.

    Parameters
    ----------
    mapped_events : pd.DataFrame
        Must contain ``timestamp``, ``activity_name``, ``source``,
        ``original_text``, ``mapped_activity``.  Optional columns:
        ``hands_count``, ``nearest_audio_activity``.

    Returns
    -------
    list[dict]
        Each dict has: timestamp, agenda_item, motion_text, hands_detected,
        outcome, source_confidence.
    """
    if mapped_events.empty:
        return []

    df = mapped_events.copy()

    # Ensure optional columns exist with safe defaults
    if "hands_count" not in df.columns:
        df["hands_count"] = 0
    df["hands_count"] = pd.to_numeric(df["hands_count"], errors="coerce").fillna(0).astype(int)

    # Build boolean mask for vote-relevant rows
    name_match = df["activity_name"].astype(str).str.contains(
        r"Vote|Motion|Confirmed Vote", case=False, na=False,
    )
    fused_source = df["source"].astype(str) == "Fused (Audio+Video)"
    has_hands = df["hands_count"] > 0

    vote_df = df[name_match | fused_source | has_hands].copy()

    if vote_df.empty:
        return []

    # Sort chronologically
    vote_df["_sec"] = vote_df["timestamp"].apply(ts_to_seconds)
    vote_df = vote_df.sort_values("_sec").reset_index(drop=True)

    records: list[dict] = []
    for _, row in vote_df.iterrows():
        text = str(row.get("original_text", ""))
        source = str(row.get("source", ""))
        records.append({
            "timestamp": str(row["timestamp"]),
            "agenda_item": str(row.get("mapped_activity", "Unknown")),
            "motion_text": text,
            "hands_detected": int(row["hands_count"]),
            "outcome": _infer_outcome(text),
            "source_confidence": _confidence_level(source),
        })

    return records


def render_voting_records(records: list) -> None:
    """Render voting records in Streamlit with summary stats.

    Parameters
    ----------
    records : list
        Output of :func:`extract_voting_records`.
    """
    if not records:
        st.info("No voting records detected in this meeting.")
        return

    st.subheader("Voting Records")

    # Summary counts
    total = len(records)
    passed = sum(1 for r in records if r["outcome"] == "passed")
    failed = sum(1 for r in records if r["outcome"] == "failed")
    unclear = total - passed - failed

    cols = st.columns(4)
    cols[0].metric("Total Votes", total)
    cols[1].metric("Passed", passed)
    cols[2].metric("Failed", failed)
    cols[3].metric("Unclear", unclear)

    # Build display dataframe
    df = pd.DataFrame(records)
    display_cols = [
        "timestamp", "agenda_item", "motion_text",
        "hands_detected", "outcome", "source_confidence",
    ]
    st.dataframe(
        df[display_cols].rename(columns={
            "timestamp": "Time",
            "agenda_item": "Agenda Item",
            "motion_text": "Motion Text",
            "hands_detected": "Hands",
            "outcome": "Outcome",
            "source_confidence": "Confidence",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ── internal helpers ─────────────────────────────────────────────────

def _infer_outcome(text: str) -> str:
    """Return 'passed', 'failed', or 'unclear' based on text patterns."""
    for pattern, label in _OUTCOME_PATTERNS:
        if pattern.search(text):
            return label
    return "unclear"


def _confidence_level(source: str) -> str:
    """Map event source to confidence tier."""
    if "Fused" in source:
        return "high"
    if "Audio" in source or "Whisper" in source or "NLP" in source:
        return "medium"
    return "low"
