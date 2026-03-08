"""Turn-taking analysis for speaker-attributed meeting events.

Provides metrics for speaker equity, interruption detection, turn-length
analysis, and dominance patterns in council meeting transcripts.

Example usage:
    from research.diarization.turn_taking import (
        compute_turn_taking_metrics,
        detect_interruptions,
        compute_turn_sequence,
    )

    metrics = compute_turn_taking_metrics(events_df)
    interruptions = detect_interruptions(events_df, gap_threshold=2.0)
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project-level time utilities
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.time_utils import ts_to_seconds, seconds_to_ts  # noqa: E402

logger = logging.getLogger(__name__)


def compute_turn_sequence(events_df: pd.DataFrame) -> list[dict]:
    """Extract the ordered sequence of speaking turns from event data.

    A "turn" is a contiguous run of events from the same speaker.
    Adjacent events by the same speaker are merged into a single turn.

    Args:
        events_df: DataFrame with 'timestamp', 'speaker' columns.
                   Optionally 'start_seconds' and 'end_seconds'.

    Returns:
        List of turn dicts::

            [
                {
                    "speaker": str,
                    "turn_index": int,
                    "start_seconds": float,
                    "end_seconds": float,
                    "duration_seconds": float,
                    "event_count": int,
                },
                ...
            ]
    """
    if events_df.empty or "speaker" not in events_df.columns:
        return []

    df = events_df.copy()

    # Ensure we have seconds-based timing
    has_precise_timing = "start_seconds" in df.columns and "end_seconds" in df.columns
    if not has_precise_timing:
        df["start_seconds"] = df["timestamp"].apply(lambda t: float(ts_to_seconds(t)))
        df["end_seconds"] = df["start_seconds"] + 5.0  # approximate 5s per event

    df = df.sort_values("start_seconds").reset_index(drop=True)

    turns: list[dict] = []
    current_speaker: Optional[str] = None
    turn_start = 0.0
    turn_end = 0.0
    event_count = 0

    for _, row in df.iterrows():
        spk = row["speaker"]
        s_start = float(row["start_seconds"])
        s_end = float(row["end_seconds"])

        if spk != current_speaker:
            # Flush previous turn
            if current_speaker is not None and current_speaker != "UNKNOWN":
                turns.append({
                    "speaker": current_speaker,
                    "turn_index": len(turns),
                    "start_seconds": round(turn_start, 2),
                    "end_seconds": round(turn_end, 2),
                    "duration_seconds": round(turn_end - turn_start, 2),
                    "event_count": event_count,
                })
            current_speaker = spk
            turn_start = s_start
            turn_end = s_end
            event_count = 1
        else:
            turn_end = max(turn_end, s_end)
            event_count += 1

    # Flush final turn
    if current_speaker is not None and current_speaker != "UNKNOWN":
        turns.append({
            "speaker": current_speaker,
            "turn_index": len(turns),
            "start_seconds": round(turn_start, 2),
            "end_seconds": round(turn_end, 2),
            "duration_seconds": round(turn_end - turn_start, 2),
            "event_count": event_count,
        })

    return turns


def compute_turn_taking_metrics(events_df: pd.DataFrame) -> dict:
    """Compute turn-taking metrics from speaker-attributed event data.

    Args:
        events_df: DataFrame with 'timestamp' and 'speaker' columns.
                   Optionally 'start_seconds' and 'end_seconds'.

    Returns:
        Dictionary containing::

            {
                "total_turns": int,
                "unique_speakers": int,
                "speaker_equity_index": float,     # 0 = perfect equity, 1 = total monopoly
                "gini_coefficient": float,          # Gini of floor-time distribution
                "avg_turn_length_seconds": float,
                "median_turn_length_seconds": float,
                "max_turn_length_seconds": float,
                "dominant_speaker": str | None,
                "dominant_speaker_share": float,    # fraction of total floor time
                "interruption_count": int,
                "turns_per_speaker": dict[str, int],
                "floor_time_per_speaker": dict[str, float],
                "per_speaker_avg_turn_length": dict[str, float],
            }
    """
    turns = compute_turn_sequence(events_df)

    if not turns:
        return {
            "total_turns": 0,
            "unique_speakers": 0,
            "speaker_equity_index": 0.0,
            "gini_coefficient": 0.0,
            "avg_turn_length_seconds": 0.0,
            "median_turn_length_seconds": 0.0,
            "max_turn_length_seconds": 0.0,
            "dominant_speaker": None,
            "dominant_speaker_share": 0.0,
            "interruption_count": 0,
            "turns_per_speaker": {},
            "floor_time_per_speaker": {},
            "per_speaker_avg_turn_length": {},
        }

    durations = [t["duration_seconds"] for t in turns]
    speakers = list({t["speaker"] for t in turns})

    # Per-speaker aggregation
    turns_per_speaker: dict[str, int] = Counter()
    floor_time_per_speaker: dict[str, float] = {}

    for spk in speakers:
        spk_turns = [t for t in turns if t["speaker"] == spk]
        turns_per_speaker[spk] = len(spk_turns)
        floor_time_per_speaker[spk] = round(
            sum(t["duration_seconds"] for t in spk_turns), 2
        )

    total_floor = sum(floor_time_per_speaker.values())

    # Per-speaker average turn length
    per_speaker_avg = {}
    for spk in speakers:
        count = turns_per_speaker[spk]
        per_speaker_avg[spk] = round(floor_time_per_speaker[spk] / count, 2) if count else 0.0

    # Dominant speaker (by floor time)
    dominant = max(floor_time_per_speaker, key=floor_time_per_speaker.get) if floor_time_per_speaker else None
    dominant_share = (
        round(floor_time_per_speaker[dominant] / total_floor, 4) if dominant and total_floor > 0 else 0.0
    )

    # Gini coefficient of floor-time distribution
    gini = _gini_coefficient(list(floor_time_per_speaker.values()))

    # Speaker equity index: normalized entropy-based measure
    # 0 = perfectly equal, 1 = single speaker dominates
    equity_index = _speaker_equity_index(list(floor_time_per_speaker.values()))

    # Interruptions: count transitions with very short gap (< 2s)
    interruptions = detect_interruptions(events_df, gap_threshold=2.0)

    return {
        "total_turns": len(turns),
        "unique_speakers": len(speakers),
        "speaker_equity_index": round(equity_index, 4),
        "gini_coefficient": round(gini, 4),
        "avg_turn_length_seconds": round(float(np.mean(durations)), 2),
        "median_turn_length_seconds": round(float(np.median(durations)), 2),
        "max_turn_length_seconds": round(float(np.max(durations)), 2),
        "dominant_speaker": dominant,
        "dominant_speaker_share": dominant_share,
        "interruption_count": len(interruptions),
        "turns_per_speaker": dict(turns_per_speaker),
        "floor_time_per_speaker": floor_time_per_speaker,
        "per_speaker_avg_turn_length": per_speaker_avg,
    }


def detect_interruptions(
    events_df: pd.DataFrame,
    gap_threshold: float = 2.0,
) -> list[dict]:
    """Identify speaker transitions that occur within a short time gap.

    When speaker B starts speaking less than *gap_threshold* seconds after
    speaker A's last utterance ends, this is flagged as a potential
    interruption.

    Args:
        events_df:     Speaker-attributed DataFrame with timing columns.
        gap_threshold: Maximum gap in seconds for a transition to be
                       considered an interruption.

    Returns:
        List of interruption dicts::

            [
                {
                    "index": int,                # sequential interruption number
                    "interrupted_speaker": str,
                    "interrupting_speaker": str,
                    "timestamp": str,            # HH:MM:SS of the interruption
                    "gap_seconds": float,        # time between end of A and start of B
                    "interrupted_turn_end": float,
                    "interrupting_turn_start": float,
                },
                ...
            ]
    """
    turns = compute_turn_sequence(events_df)

    if len(turns) < 2:
        return []

    interruptions: list[dict] = []

    for i in range(1, len(turns)):
        prev_turn = turns[i - 1]
        curr_turn = turns[i]

        # Must be different speakers
        if prev_turn["speaker"] == curr_turn["speaker"]:
            continue

        gap = curr_turn["start_seconds"] - prev_turn["end_seconds"]

        # Negative gap means overlap (definite interruption);
        # small positive gap means rapid transition (likely interruption)
        if gap < gap_threshold:
            interruptions.append({
                "index": len(interruptions),
                "interrupted_speaker": prev_turn["speaker"],
                "interrupting_speaker": curr_turn["speaker"],
                "timestamp": seconds_to_ts(int(curr_turn["start_seconds"])),
                "gap_seconds": round(gap, 2),
                "interrupted_turn_end": prev_turn["end_seconds"],
                "interrupting_turn_start": curr_turn["start_seconds"],
            })

    return interruptions


def detect_long_monologues(
    events_df: pd.DataFrame,
    threshold_seconds: float = 120.0,
) -> list[dict]:
    """Identify speaking turns that exceed a duration threshold.

    Useful for finding presentations, filibusters, or extended public comments.

    Args:
        events_df:         Speaker-attributed DataFrame.
        threshold_seconds: Minimum duration (in seconds) for a turn to be
                           considered a "long monologue".

    Returns:
        List of monologue dicts with speaker, start, end, and duration.
    """
    turns = compute_turn_sequence(events_df)
    monologues = []

    for turn in turns:
        if turn["duration_seconds"] >= threshold_seconds:
            monologues.append({
                "speaker": turn["speaker"],
                "start_ts": seconds_to_ts(int(turn["start_seconds"])),
                "end_ts": seconds_to_ts(int(turn["end_seconds"])),
                "duration_seconds": turn["duration_seconds"],
                "event_count": turn["event_count"],
            })

    return monologues


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient for a list of non-negative values.

    Returns 0.0 for perfect equality, approaching 1.0 for total inequality.
    Returns 0.0 for empty or all-zero inputs.
    """
    if not values or all(v == 0 for v in values):
        return 0.0

    arr = np.array(sorted(values), dtype=float)
    n = len(arr)
    total = arr.sum()

    if total == 0:
        return 0.0

    # Standard Gini formula
    cumulative = np.cumsum(arr)
    gini = (2.0 * np.sum((np.arange(1, n + 1) * arr))) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, gini))


def _speaker_equity_index(floor_times: list[float]) -> float:
    """Compute a normalized equity index from floor-time distribution.

    Based on deviation from perfect equity (uniform distribution).
    0 = all speakers have equal floor time.
    1 = single speaker has all floor time.

    Uses 1 - (H / H_max) where H is Shannon entropy and H_max = log(n).
    """
    if not floor_times or all(t == 0 for t in floor_times):
        return 0.0

    total = sum(floor_times)
    if total == 0:
        return 0.0

    n = len(floor_times)
    if n <= 1:
        return 0.0

    # Shannon entropy
    probs = [t / total for t in floor_times if t > 0]
    h = -sum(p * np.log2(p) for p in probs if p > 0)
    h_max = np.log2(n)

    if h_max == 0:
        return 0.0

    return 1.0 - (h / h_max)
