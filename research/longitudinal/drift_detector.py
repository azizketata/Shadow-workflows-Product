"""Concept drift detection for meeting process evolution.

Analyzes how meeting processes change across multiple meetings using
pm4py temporal profiling, statistical change detection, and shadow
workflow frequency tracking.

Integrates with:
- features.trends (SQLite meeting_history.db) for historical data
- pm4py temporal profiling for process-level drift detection
- pipeline.time_utils for timestamp conversions
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fitness trend analysis
# ---------------------------------------------------------------------------


def compute_fitness_trend(meeting_results: list[dict]) -> dict:
    """Analyze fitness score progression across meetings.

    Uses linear regression on dedup fitness scores to detect whether
    meeting compliance is improving, stable, or declining. Also identifies
    change points where fitness shifts significantly.

    Args:
        meeting_results: List of dicts, each with keys:
            - name (str): Meeting identifier
            - date (str): Date in YYYY-MM-DD format
            - fitness_dedup (float): Deduplicated fitness percentage
            - fitness_raw (float): Raw fitness percentage
            - agenda_coverage (float): Percentage of agenda items covered
            - shadow_count (int): Number of shadow events
            - event_count (int): Total event count

    Returns:
        Dictionary with:
        - trend_direction: "improving", "declining", or "stable"
        - slope: Float slope of the linear regression on fitness_dedup
        - intercept: Float intercept of the regression
        - r_squared: Float R-squared value (goodness of fit)
        - change_points: List of dicts identifying meetings where fitness
            shifted significantly (>10% absolute change)
        - regression_stats: Dict with detailed regression information
        - meeting_count: Number of meetings analyzed
        - fitness_range: (min, max) tuple of fitness_dedup values
    """
    if not meeting_results:
        return _empty_fitness_trend()

    # Sort by date
    sorted_results = sorted(
        meeting_results,
        key=lambda r: r.get("date", "1970-01-01"),
    )

    n = len(sorted_results)
    if n < 2:
        fitness = sorted_results[0].get("fitness_dedup", 0.0)
        return {
            "trend_direction": "stable",
            "slope": 0.0,
            "intercept": fitness,
            "r_squared": 0.0,
            "change_points": [],
            "regression_stats": {
                "n": 1,
                "mean_fitness": fitness,
                "std_fitness": 0.0,
            },
            "meeting_count": 1,
            "fitness_range": (fitness, fitness),
        }

    # Extract fitness values
    x = np.arange(n, dtype=float)
    y = np.array(
        [r.get("fitness_dedup", 0.0) for r in sorted_results], dtype=float
    )

    # Linear regression via least squares
    slope, intercept, r_squared = _linear_regression(x, y)

    # Determine trend direction
    if abs(slope) < 1.0:  # Less than 1% per meeting
        trend_direction = "stable"
    elif slope > 0:
        trend_direction = "improving"
    else:
        trend_direction = "declining"

    # Detect change points (>10% absolute shift between consecutive meetings)
    change_points = []
    for i in range(1, n):
        prev_fitness = sorted_results[i - 1].get("fitness_dedup", 0.0)
        curr_fitness = sorted_results[i].get("fitness_dedup", 0.0)
        delta = curr_fitness - prev_fitness

        if abs(delta) > 10.0:
            change_points.append({
                "meeting_index": i,
                "meeting_name": sorted_results[i].get("name", f"Meeting_{i}"),
                "date": sorted_results[i].get("date", ""),
                "previous_fitness": round(prev_fitness, 2),
                "current_fitness": round(curr_fitness, 2),
                "delta": round(delta, 2),
                "direction": "improvement" if delta > 0 else "decline",
            })

    return {
        "trend_direction": trend_direction,
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "change_points": change_points,
        "regression_stats": {
            "n": n,
            "mean_fitness": round(float(np.mean(y)), 2),
            "std_fitness": round(float(np.std(y)), 2),
            "min_fitness": round(float(np.min(y)), 2),
            "max_fitness": round(float(np.max(y)), 2),
        },
        "meeting_count": n,
        "fitness_range": (round(float(np.min(y)), 2), round(float(np.max(y)), 2)),
    }


# ---------------------------------------------------------------------------
# Shadow workflow evolution tracking
# ---------------------------------------------------------------------------


def track_shadow_evolution(meeting_shadows: list[dict]) -> dict:
    """Track shadow workflow frequency and persistence across meetings.

    Categorizes shadow labels into four groups based on their appearance
    frequency across the set of meetings:
    - persistent: Appear in >50% of meetings (structural patterns)
    - emerging: Appear only in the latest 30% of meetings
    - declining: Appear only in the earliest 30% of meetings
    - one_off: Appear in exactly 1 meeting

    Args:
        meeting_shadows: List of dicts, each with:
            - meeting_name (str): Meeting identifier
            - shadows (list[str]): List of shadow workflow labels

    Returns:
        Dictionary with:
        - persistent_shadows: List of labels appearing in >50% of meetings
        - emerging_shadows: Labels appearing only in recent meetings
        - declining_shadows: Labels appearing only in early meetings
        - one_off_shadows: Labels appearing in exactly one meeting
        - shadow_frequency: Dict mapping label -> count of meetings
        - total_unique_shadows: Number of distinct shadow labels
        - meeting_count: Number of meetings analyzed
        - persistence_matrix: Dict mapping label -> list of meeting names
    """
    if not meeting_shadows:
        return _empty_shadow_evolution()

    n_meetings = len(meeting_shadows)
    if n_meetings < 1:
        return _empty_shadow_evolution()

    # Build label -> set of meeting indices
    label_meetings: dict[str, set[int]] = defaultdict(set)
    label_meeting_names: dict[str, list[str]] = defaultdict(list)

    for i, entry in enumerate(meeting_shadows):
        meeting_name = entry.get("meeting_name", f"Meeting_{i}")
        shadows = entry.get("shadows", [])
        unique_shadows = set(shadows)  # Deduplicate within a meeting
        for label in unique_shadows:
            label_meetings[label].add(i)
            label_meeting_names[label].append(meeting_name)

    # Classify each shadow label
    persistent = []
    emerging = []
    declining = []
    one_off = []

    threshold_persistent = n_meetings * 0.5
    # "Recent" = last 30% of meetings (at least 1)
    recent_start = max(0, n_meetings - max(1, int(n_meetings * 0.3)))
    recent_indices = set(range(recent_start, n_meetings))
    # "Early" = first 30% of meetings (at least 1)
    early_end = min(n_meetings, max(1, int(n_meetings * 0.3)))
    early_indices = set(range(0, early_end))

    shadow_frequency: dict[str, int] = {}

    for label, indices in label_meetings.items():
        count = len(indices)
        shadow_frequency[label] = count

        if count == 1:
            one_off.append(label)
        elif count > threshold_persistent:
            persistent.append(label)
        elif indices.issubset(recent_indices):
            emerging.append(label)
        elif indices.issubset(early_indices):
            declining.append(label)
        # Labels that don't fit neatly are just tracked in frequency

    # Sort by frequency (descending)
    persistent.sort(key=lambda l: -shadow_frequency.get(l, 0))
    emerging.sort(key=lambda l: -shadow_frequency.get(l, 0))
    declining.sort(key=lambda l: -shadow_frequency.get(l, 0))
    one_off.sort()

    return {
        "persistent_shadows": persistent,
        "emerging_shadows": emerging,
        "declining_shadows": declining,
        "one_off_shadows": one_off,
        "shadow_frequency": dict(
            sorted(shadow_frequency.items(), key=lambda x: -x[1])
        ),
        "total_unique_shadows": len(label_meetings),
        "meeting_count": n_meetings,
        "persistence_matrix": dict(label_meeting_names),
    }


# ---------------------------------------------------------------------------
# Concept drift detection using pm4py temporal profiles
# ---------------------------------------------------------------------------


def detect_concept_drift(
    meeting_logs: list[pd.DataFrame],
    meeting_dates: list[str],
    zeta: float = 2.0,
) -> dict:
    """Detect process changes using temporal profile deviations.

    For each meeting log, discovers a temporal profile (expected duration
    between activity pairs), then checks subsequent meetings against that
    baseline to find significant timing deviations.

    Uses pm4py.discover_temporal_profile and
    pm4py.conformance_temporal_profile.

    Args:
        meeting_logs: List of pm4py-compatible DataFrames (must have
            'case:concept:name', 'concept:name', 'time:timestamp' columns,
            or will be auto-converted from project format).
        meeting_dates: Parallel list of date strings (YYYY-MM-DD).
        zeta: Z-score threshold for temporal profile deviation.
            Default 2.0 (95% confidence).

    Returns:
        Dictionary with:
        - drift_detected: bool
        - drift_points: List of dicts with meeting pairs showing drift
        - temporal_deviations: List of activity pairs with timing changes
        - baseline_meeting: Name/date of the baseline meeting
        - summary: Human-readable summary string
    """
    if len(meeting_logs) < 2:
        return {
            "drift_detected": False,
            "drift_points": [],
            "temporal_deviations": [],
            "baseline_meeting": meeting_dates[0] if meeting_dates else "",
            "summary": "Insufficient data: need at least 2 meetings for drift detection.",
        }

    try:
        import pm4py
    except ImportError:
        logger.error("pm4py not available; cannot perform temporal drift detection.")
        return {
            "drift_detected": False,
            "drift_points": [],
            "temporal_deviations": [],
            "baseline_meeting": "",
            "summary": "pm4py library not available.",
        }

    # Ensure logs are in pm4py format
    prepared_logs = []
    for i, log_df in enumerate(meeting_logs):
        prepared = _prepare_for_pm4py(log_df, case_id=f"meeting_{i}")
        if prepared is not None and not prepared.empty:
            prepared_logs.append(prepared)
        else:
            logger.warning(
                "Meeting log %d (%s) could not be prepared; skipping.",
                i,
                meeting_dates[i] if i < len(meeting_dates) else "unknown",
            )

    if len(prepared_logs) < 2:
        return {
            "drift_detected": False,
            "drift_points": [],
            "temporal_deviations": [],
            "baseline_meeting": meeting_dates[0] if meeting_dates else "",
            "summary": "Insufficient valid logs after preparation.",
        }

    # Use first meeting as baseline
    baseline_log = prepared_logs[0]
    baseline_date = meeting_dates[0] if meeting_dates else "baseline"

    try:
        temporal_profile = pm4py.discover_temporal_profile(baseline_log)
    except Exception as exc:
        logger.warning("Failed to discover temporal profile: %s", exc)
        return {
            "drift_detected": False,
            "drift_points": [],
            "temporal_deviations": [],
            "baseline_meeting": baseline_date,
            "summary": f"Temporal profile discovery failed: {exc}",
        }

    # Check each subsequent meeting against baseline
    drift_points = []
    all_deviations: list[dict] = []

    for i in range(1, len(prepared_logs)):
        date_i = meeting_dates[i] if i < len(meeting_dates) else f"meeting_{i}"

        try:
            conformance = pm4py.conformance_temporal_profile(
                prepared_logs[i],
                temporal_profile,
                zeta=zeta,
            )
        except Exception as exc:
            logger.warning(
                "Temporal conformance check failed for meeting %d: %s", i, exc
            )
            continue

        # Extract deviations from conformance result
        meeting_deviations = []
        for case_result in conformance:
            if isinstance(case_result, list):
                for deviation in case_result:
                    if isinstance(deviation, tuple) and len(deviation) >= 4:
                        activity_pair = deviation[0] if len(deviation) > 0 else "?"
                        observed = deviation[1] if len(deviation) > 1 else 0
                        expected_mean = deviation[2] if len(deviation) > 2 else 0
                        expected_std = deviation[3] if len(deviation) > 3 else 0
                        meeting_deviations.append({
                            "activity_pair": str(activity_pair),
                            "observed_duration": observed,
                            "expected_mean": expected_mean,
                            "expected_std": expected_std,
                            "z_score": (
                                abs(observed - expected_mean) / expected_std
                                if expected_std > 0
                                else 0
                            ),
                        })
            elif isinstance(case_result, dict):
                # Some pm4py versions return dicts
                deviations_list = case_result.get("deviations", [])
                for dev in deviations_list:
                    meeting_deviations.append({
                        "activity_pair": str(dev.get("activity_pair", "?")),
                        "observed_duration": dev.get("observed", 0),
                        "expected_mean": dev.get("expected_mean", 0),
                        "expected_std": dev.get("expected_std", 0),
                        "z_score": dev.get("z_score", 0),
                    })

        if meeting_deviations:
            drift_points.append({
                "meeting_index": i,
                "meeting_date": date_i,
                "deviation_count": len(meeting_deviations),
                "max_z_score": max(d["z_score"] for d in meeting_deviations),
            })
            all_deviations.extend(meeting_deviations)

    drift_detected = len(drift_points) > 0

    # Deduplicate deviations by activity pair, keeping the one with highest z-score
    unique_deviations: dict[str, dict] = {}
    for dev in all_deviations:
        pair = dev["activity_pair"]
        if pair not in unique_deviations or dev["z_score"] > unique_deviations[pair]["z_score"]:
            unique_deviations[pair] = dev

    sorted_deviations = sorted(
        unique_deviations.values(), key=lambda d: -d["z_score"]
    )

    # Summary
    if drift_detected:
        summary = (
            f"Concept drift detected in {len(drift_points)} of "
            f"{len(prepared_logs) - 1} meetings compared to baseline "
            f"({baseline_date}). {len(sorted_deviations)} unique activity "
            f"pair timing deviations found (zeta={zeta})."
        )
    else:
        summary = (
            f"No significant concept drift detected across "
            f"{len(prepared_logs)} meetings (zeta={zeta})."
        )

    return {
        "drift_detected": drift_detected,
        "drift_points": drift_points,
        "temporal_deviations": sorted_deviations[:20],  # Top 20
        "baseline_meeting": baseline_date,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _linear_regression(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float]:
    """Simple linear regression returning (slope, intercept, r_squared).

    Uses numpy for a lightweight least-squares fit without requiring scipy.
    """
    n = len(x)
    if n < 2:
        return 0.0, float(y[0]) if n == 1 else 0.0, 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((y - y_mean) ** 2)

    if ss_xx == 0:
        return 0.0, float(y_mean), 0.0

    slope = float(ss_xy / ss_xx)
    intercept = float(y_mean - slope * x_mean)

    if ss_yy == 0:
        r_squared = 1.0  # Perfect fit (all y values identical)
    else:
        r_squared = float((ss_xy ** 2) / (ss_xx * ss_yy))

    return slope, intercept, r_squared


def _prepare_for_pm4py(
    df: pd.DataFrame, case_id: str = "meeting_1"
) -> pd.DataFrame | None:
    """Convert a project-format DataFrame to pm4py event log format.

    Handles the column mapping from the project's convention
    (timestamp, activity_name, source, details) to pm4py's expected
    columns (case:concept:name, concept:name, time:timestamp).

    Args:
        df: Input DataFrame in project format.
        case_id: Case identifier to assign.

    Returns:
        pm4py-compatible DataFrame, or None if conversion fails.
    """
    if df is None or df.empty:
        return None

    result = pd.DataFrame()

    # Case ID
    if "case:concept:name" in df.columns:
        result["case:concept:name"] = df["case:concept:name"]
    elif "case_id" in df.columns:
        result["case:concept:name"] = df["case_id"]
    else:
        result["case:concept:name"] = case_id

    # Activity name
    if "concept:name" in df.columns:
        result["concept:name"] = df["concept:name"]
    elif "mapped_activity" in df.columns:
        result["concept:name"] = df["mapped_activity"]
    elif "activity_name" in df.columns:
        result["concept:name"] = df["activity_name"]
    else:
        logger.warning("No activity column found in DataFrame.")
        return None

    # Timestamp
    if "time:timestamp" in df.columns:
        result["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    elif "timestamp" in df.columns:
        # Convert HH:MM:SS to datetime anchored at a reference date
        from pipeline.time_utils import ts_to_seconds

        base_date = datetime(2026, 1, 1)
        seconds = df["timestamp"].apply(ts_to_seconds)
        result["time:timestamp"] = pd.to_datetime(
            seconds.apply(
                lambda s: base_date + pd.Timedelta(seconds=int(s))
            )
        )
    else:
        # Generate sequential timestamps
        base_date = datetime(2026, 1, 1)
        result["time:timestamp"] = pd.to_datetime(
            [base_date + pd.Timedelta(seconds=i * 60) for i in range(len(df))]
        )

    return result


def _empty_fitness_trend() -> dict:
    """Return an empty fitness trend result."""
    return {
        "trend_direction": "stable",
        "slope": 0.0,
        "intercept": 0.0,
        "r_squared": 0.0,
        "change_points": [],
        "regression_stats": {"n": 0, "mean_fitness": 0.0, "std_fitness": 0.0},
        "meeting_count": 0,
        "fitness_range": (0.0, 0.0),
    }


def _empty_shadow_evolution() -> dict:
    """Return an empty shadow evolution result."""
    return {
        "persistent_shadows": [],
        "emerging_shadows": [],
        "declining_shadows": [],
        "one_off_shadows": [],
        "shadow_frequency": {},
        "total_unique_shadows": 0,
        "meeting_count": 0,
        "persistence_matrix": {},
    }
