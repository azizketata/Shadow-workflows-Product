"""Social network analysis for speaker-attributed meeting events.

Uses pm4py's organizational mining functions to discover handover-of-work
and working-together networks from speaker-attributed event logs.

Example usage:
    from research.diarization.social_network import (
        build_speaker_event_log,
        discover_handover_network,
        discover_working_together,
        network_to_adjacency_matrix,
    )

    pm4py_log = build_speaker_event_log(events_df)
    handover = discover_handover_network(events_df)
    collab = discover_working_together(events_df)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project-level time utilities
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.time_utils import ts_to_seconds  # noqa: E402

# ---------------------------------------------------------------------------
# pm4py import (should be available per project requirements)
# ---------------------------------------------------------------------------
try:
    import pm4py
    from pm4py.objects.log.obj import EventLog, Trace, Event
    from pm4py.algo.organizational_mining.sna import algorithm as sna_algorithm
    _PM4PY_AVAILABLE = True
except ImportError as exc:
    _PM4PY_AVAILABLE = False
    _PM4PY_ERROR = (
        f"pm4py is not installed or failed to import. Install it with:\n"
        f"  pip install pm4py\n"
        f"Original error: {exc}"
    )

logger = logging.getLogger(__name__)


def build_speaker_event_log(events_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a speaker-attributed event DataFrame to pm4py-compatible format.

    pm4py expects at minimum these columns:
        - ``case:concept:name`` -- case identifier (all events belong to one meeting)
        - ``concept:name``      -- activity name
        - ``time:timestamp``    -- datetime timestamp
        - ``org:resource``      -- the resource (speaker) who performed the activity

    Args:
        events_df: DataFrame with at least 'timestamp', 'activity_name', and
                   'speaker' columns.

    Returns:
        A new DataFrame formatted for pm4py with the required columns.

    Raises:
        ValueError: If 'speaker' column is missing from the input DataFrame.
    """
    if "speaker" not in events_df.columns:
        raise ValueError(
            "Input DataFrame must have a 'speaker' column. "
            "Run merge_speaker_labels() or MeetingDiarizer.segments_to_dataframe() first."
        )

    df = events_df.copy()

    # Convert timestamps to datetime for pm4py (using a base date of 2026-01-01)
    base_date = pd.Timestamp("2026-01-01")
    df["_seconds"] = df["timestamp"].apply(ts_to_seconds)
    df["time:timestamp"] = df["_seconds"].apply(
        lambda s: base_date + pd.Timedelta(seconds=int(s))
    )

    # Map to pm4py column names
    df["case:concept:name"] = "meeting_001"  # single case for one meeting
    df["concept:name"] = df["activity_name"]
    df["org:resource"] = df["speaker"]

    # Sort by timestamp (pm4py expects chronological order)
    df = df.sort_values("time:timestamp").reset_index(drop=True)

    # Keep relevant columns plus originals for reference
    keep_cols = [
        "case:concept:name", "concept:name", "time:timestamp", "org:resource",
        "timestamp", "activity_name", "speaker", "source",
    ]
    extra = [c for c in ["details", "original_text", "mapped_activity"] if c in df.columns]
    keep_cols.extend(extra)

    return df[[c for c in keep_cols if c in df.columns]]


def discover_handover_network(
    events_df: pd.DataFrame,
    case_id: str = "meeting_001",
) -> dict:
    """Discover the handover-of-work social network using pm4py.

    The handover-of-work network captures how often one speaker's activity
    is directly followed by another speaker's activity, revealing sequential
    interaction patterns (e.g., Mayor -> Council Member during motions).

    Args:
        events_df: Speaker-attributed event DataFrame (must have 'speaker' column).
        case_id:   Case identifier to assign (default: 'meeting_001').

    Returns:
        Dictionary with:
            - ``matrix``:    dict of dict mapping (speaker_a, speaker_b) -> weight.
            - ``speakers``:  Sorted list of unique speakers.
            - ``edges``:     List of (source, target, weight) tuples, sorted by weight desc.
            - ``top_handovers``: Top 10 most frequent speaker transitions.

    Raises:
        ImportError: If pm4py is not available.
    """
    if not _PM4PY_AVAILABLE:
        raise ImportError(_PM4PY_ERROR)

    pm4py_df = build_speaker_event_log(events_df)

    # pm4py SNA expects an event log
    log = pm4py.convert_to_event_log(pm4py_df)

    # Discover handover-of-work network
    hw_values = sna_algorithm.apply(
        log,
        variant=sna_algorithm.Variants.HANDOVER_OF_WORK,
    )

    return _parse_sna_result(hw_values, "handover")


def discover_working_together(
    events_df: pd.DataFrame,
    case_id: str = "meeting_001",
) -> dict:
    """Discover the working-together social network using pm4py.

    The working-together network captures how often two speakers participate
    in the same case (or activity window), revealing collaboration patterns.

    Args:
        events_df: Speaker-attributed event DataFrame (must have 'speaker' column).
        case_id:   Case identifier to assign.

    Returns:
        Dictionary with the same structure as :func:`discover_handover_network`.

    Raises:
        ImportError: If pm4py is not available.
    """
    if not _PM4PY_AVAILABLE:
        raise ImportError(_PM4PY_ERROR)

    pm4py_df = build_speaker_event_log(events_df)
    log = pm4py.convert_to_event_log(pm4py_df)

    wt_values = sna_algorithm.apply(
        log,
        variant=sna_algorithm.Variants.WORKING_TOGETHER,
    )

    return _parse_sna_result(wt_values, "working_together")


def discover_subcontracting(events_df: pd.DataFrame) -> dict:
    """Discover the subcontracting social network using pm4py.

    Subcontracting captures patterns where speaker A delegates to speaker B,
    who then returns control to speaker A (A -> B -> A pattern).

    Args:
        events_df: Speaker-attributed event DataFrame.

    Returns:
        Dictionary with the same structure as :func:`discover_handover_network`.

    Raises:
        ImportError: If pm4py is not available.
    """
    if not _PM4PY_AVAILABLE:
        raise ImportError(_PM4PY_ERROR)

    pm4py_df = build_speaker_event_log(events_df)
    log = pm4py.convert_to_event_log(pm4py_df)

    sc_values = sna_algorithm.apply(
        log,
        variant=sna_algorithm.Variants.SUBCONTRACTING,
    )

    return _parse_sna_result(sc_values, "subcontracting")


def network_to_adjacency_matrix(network_result: dict) -> pd.DataFrame:
    """Convert a network result dict to a pandas adjacency matrix.

    Args:
        network_result: Output from one of the discover_* functions.

    Returns:
        Square DataFrame indexed and columned by speaker labels,
        with edge weights as values (0 where no connection exists).
    """
    speakers = network_result.get("speakers", [])
    matrix = network_result.get("matrix", {})

    adj = pd.DataFrame(0.0, index=speakers, columns=speakers)
    for src in matrix:
        for tgt in matrix[src]:
            if src in adj.index and tgt in adj.columns:
                adj.loc[src, tgt] = matrix[src][tgt]

    return adj


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_sna_result(sna_result, network_type: str) -> dict:
    """Parse pm4py SNA result into a consistent dictionary format.

    pm4py SNA results can come in different formats depending on the version
    (numpy array + labels, or dict-of-dicts). This function normalizes them.
    """
    matrix: dict[str, dict[str, float]] = {}
    speakers: list[str] = []
    edges: list[tuple[str, str, float]] = []

    try:
        # pm4py >= 2.7: SNA returns a tuple of (matrix_values, activities_list)
        # or a pandas DataFrame, depending on variant and version.
        if isinstance(sna_result, tuple) and len(sna_result) == 2:
            values, labels = sna_result
            speakers = list(labels) if not isinstance(labels, list) else labels

            # values might be a numpy array or list of lists
            import numpy as np
            if hasattr(values, 'shape'):
                arr = values
            else:
                arr = np.array(values)

            for i, src in enumerate(speakers):
                matrix[src] = {}
                for j, tgt in enumerate(speakers):
                    weight = float(arr[i][j]) if i < arr.shape[0] and j < arr.shape[1] else 0.0
                    if weight > 0:
                        matrix[src][tgt] = weight
                        edges.append((src, tgt, weight))

        elif isinstance(sna_result, dict):
            # Older pm4py format or custom dict
            for src, targets in sna_result.items():
                src_str = str(src)
                if src_str not in speakers:
                    speakers.append(src_str)
                matrix[src_str] = {}
                for tgt, weight in targets.items():
                    tgt_str = str(tgt)
                    if tgt_str not in speakers:
                        speakers.append(tgt_str)
                    w = float(weight)
                    if w > 0:
                        matrix[src_str][tgt_str] = w
                        edges.append((src_str, tgt_str, w))

        elif isinstance(sna_result, pd.DataFrame):
            speakers = list(sna_result.index)
            for src in speakers:
                matrix[src] = {}
                for tgt in speakers:
                    w = float(sna_result.loc[src, tgt])
                    if w > 0:
                        matrix[src][tgt] = w
                        edges.append((src, tgt, w))

        else:
            logger.warning("Unexpected SNA result type: %s", type(sna_result))

    except Exception as e:
        logger.error("Error parsing SNA result for %s: %s", network_type, e)

    # Sort edges by weight descending
    edges.sort(key=lambda x: x[2], reverse=True)
    speakers = sorted(set(speakers))

    return {
        "network_type": network_type,
        "matrix": matrix,
        "speakers": speakers,
        "edges": edges,
        "top_handovers": edges[:10],
    }
