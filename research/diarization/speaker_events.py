"""Merge speaker diarization labels with existing event DataFrames.

Provides temporal alignment between diarized speaker segments and the
standard Meeting Process Twin event DataFrame, plus per-speaker statistics.

Example usage:
    from research.diarization.speaker_events import merge_speaker_labels, compute_speaker_statistics

    enriched_df = merge_speaker_labels(events_df, diarization_segments)
    stats = compute_speaker_statistics(enriched_df)
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project-level time utilities
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.time_utils import ts_to_seconds, seconds_to_ts  # noqa: E402


def merge_speaker_labels(
    events_df: pd.DataFrame,
    diarization_segments: list[dict],
    max_gap_seconds: float = 2.0,
) -> pd.DataFrame:
    """Merge speaker labels into an existing event DataFrame by temporal alignment.

    For each event row, finds the diarization segment whose time interval best
    overlaps with the event timestamp.  If the closest segment is within
    *max_gap_seconds* of the event, its speaker label is assigned; otherwise
    the event is labelled ``'UNKNOWN'``.

    Args:
        events_df:              Standard pipeline DataFrame with a 'timestamp'
                                column (HH:MM:SS strings or numeric seconds).
        diarization_segments:   List of segment dicts from
                                :class:`~research.diarization.diarizer.MeetingDiarizer`.
                                Each dict must have 'start', 'end', and 'speaker'.
        max_gap_seconds:        Maximum allowable gap (in seconds) between an
                                event timestamp and the nearest diarization
                                segment for a match to be accepted.

    Returns:
        A copy of *events_df* with two new columns:
            - ``speaker``: Assigned speaker label (str).
            - ``speaker_confidence``: ``'overlap'`` if the event falls inside a
              segment, ``'nearest'`` if matched by proximity, or ``'none'``
              if no segment was close enough.
    """
    if events_df.empty or not diarization_segments:
        result = events_df.copy()
        result["speaker"] = "UNKNOWN"
        result["speaker_confidence"] = "none"
        return result

    # Pre-process diarization segments into a sorted list for fast lookup
    sorted_segs = sorted(diarization_segments, key=lambda s: s.get("start", 0.0))
    seg_starts = [s.get("start", 0.0) for s in sorted_segs]
    seg_ends = [s.get("end", 0.0) for s in sorted_segs]
    seg_speakers = [s.get("speaker", "UNKNOWN") for s in sorted_segs]

    def _find_speaker(event_ts_seconds: float) -> tuple[str, str]:
        """Return (speaker, confidence) for a given timestamp in seconds."""
        best_speaker = "UNKNOWN"
        best_confidence = "none"
        best_distance = float("inf")

        for i, (start, end, spk) in enumerate(zip(seg_starts, seg_ends, seg_speakers)):
            # Check for direct overlap
            if start <= event_ts_seconds <= end:
                return spk, "overlap"

            # Compute distance to segment boundary
            if event_ts_seconds < start:
                dist = start - event_ts_seconds
            else:
                dist = event_ts_seconds - end

            if dist < best_distance:
                best_distance = dist
                best_speaker = spk
                best_confidence = "nearest"

        if best_distance <= max_gap_seconds:
            return best_speaker, best_confidence
        return "UNKNOWN", "none"

    # Convert event timestamps to seconds
    result = events_df.copy()
    event_seconds = result["timestamp"].apply(ts_to_seconds)

    speakers = []
    confidences = []
    for ts_sec in event_seconds:
        spk, conf = _find_speaker(float(ts_sec))
        speakers.append(spk)
        confidences.append(conf)

    result["speaker"] = speakers
    result["speaker_confidence"] = confidences
    return result


def compute_speaker_statistics(events_df: pd.DataFrame) -> dict:
    """Compute per-speaker statistics from a speaker-attributed event DataFrame.

    Expects the DataFrame to have a ``speaker`` column (as produced by
    :func:`merge_speaker_labels` or :meth:`MeetingDiarizer.segments_to_dataframe`).

    Args:
        events_df: DataFrame with at least 'timestamp', 'speaker', and
                   'activity_name' columns.  Optional: 'start_seconds',
                   'end_seconds' for floor-time calculations.

    Returns:
        Dictionary with the following structure::

            {
                "total_events": int,
                "unique_speakers": int,
                "speaker_list": list[str],
                "per_speaker": {
                    "SPEAKER_00": {
                        "event_count": int,
                        "event_share": float,          # fraction of total events
                        "floor_time_seconds": float,   # total speaking time (if available)
                        "floor_time_share": float,     # fraction of total floor time
                        "activity_distribution": dict,  # {activity_name: count}
                        "first_event_ts": str,          # HH:MM:SS
                        "last_event_ts": str,           # HH:MM:SS
                        "interventions": int,           # number of distinct speaking turns
                    },
                    ...
                },
                "most_active_speaker": str,
                "least_active_speaker": str,
            }
    """
    if events_df.empty or "speaker" not in events_df.columns:
        return {
            "total_events": 0,
            "unique_speakers": 0,
            "speaker_list": [],
            "per_speaker": {},
            "most_active_speaker": None,
            "least_active_speaker": None,
        }

    df = events_df.copy()

    # Ensure we have seconds for floor-time calculation
    has_timing = "start_seconds" in df.columns and "end_seconds" in df.columns
    if not has_timing:
        # Fall back to timestamp parsing
        df["_event_seconds"] = df["timestamp"].apply(ts_to_seconds)

    speakers = [s for s in df["speaker"].unique() if s != "UNKNOWN"]
    total_events = len(df)
    total_floor_time = 0.0

    per_speaker: dict[str, dict] = {}

    for spk in speakers:
        spk_df = df[df["speaker"] == spk]
        event_count = len(spk_df)

        # Floor time
        if has_timing:
            floor_time = (spk_df["end_seconds"] - spk_df["start_seconds"]).clip(lower=0).sum()
        else:
            # Approximate: assume each event spans until the next event or +5 seconds
            sorted_times = spk_df["_event_seconds"].sort_values().tolist()
            floor_time = 0.0
            for i, t in enumerate(sorted_times):
                if i + 1 < len(sorted_times):
                    gap = sorted_times[i + 1] - t
                    floor_time += min(gap, 30.0)  # cap at 30s per event
                else:
                    floor_time += 5.0  # last event: assume 5s
        total_floor_time += floor_time

        # Activity distribution
        activity_dist = Counter(spk_df["activity_name"].tolist())

        # First/last event timestamps
        ts_seconds = spk_df["timestamp"].apply(ts_to_seconds)
        first_ts = seconds_to_ts(ts_seconds.min()) if not ts_seconds.empty else "00:00:00"
        last_ts = seconds_to_ts(ts_seconds.max()) if not ts_seconds.empty else "00:00:00"

        # Interventions: count contiguous runs of this speaker
        interventions = _count_interventions(df, spk)

        per_speaker[spk] = {
            "event_count": event_count,
            "event_share": round(event_count / total_events, 4) if total_events else 0.0,
            "floor_time_seconds": round(floor_time, 2),
            "floor_time_share": 0.0,  # computed after loop
            "activity_distribution": dict(activity_dist),
            "first_event_ts": first_ts,
            "last_event_ts": last_ts,
            "interventions": interventions,
        }

    # Compute floor-time shares
    if total_floor_time > 0:
        for spk in per_speaker:
            per_speaker[spk]["floor_time_share"] = round(
                per_speaker[spk]["floor_time_seconds"] / total_floor_time, 4
            )

    # Most/least active
    most_active = max(per_speaker, key=lambda s: per_speaker[s]["event_count"]) if per_speaker else None
    least_active = min(per_speaker, key=lambda s: per_speaker[s]["event_count"]) if per_speaker else None

    return {
        "total_events": total_events,
        "unique_speakers": len(speakers),
        "speaker_list": sorted(speakers),
        "per_speaker": per_speaker,
        "most_active_speaker": most_active,
        "least_active_speaker": least_active,
    }


def _count_interventions(df: pd.DataFrame, speaker: str) -> int:
    """Count the number of distinct speaking turns for a given speaker.

    A new intervention starts whenever the speaker column transitions
    from a different speaker to the target speaker.
    """
    speaker_col = df["speaker"].tolist()
    interventions = 0
    prev_was_this_speaker = False

    for spk in speaker_col:
        if spk == speaker:
            if not prev_was_this_speaker:
                interventions += 1
            prev_was_this_speaker = True
        else:
            prev_was_this_speaker = False

    return interventions
