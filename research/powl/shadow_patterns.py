"""Shadow workflow pattern detection and classification.

Groups co-occurring shadow/deviation events into temporal clusters and
classifies them into structural patterns: concurrent, sequential,
recurring, or isolated.

This analysis reveals whether shadow workflows are:
- Isolated one-offs (noise or ad-hoc)
- Recurring patterns (systematic informal practices)
- Concurrent with formal activities (parallel shadow processes)
- Sequential chains (multi-step informal procedures)
"""

import sys
import os
from collections import Counter, defaultdict

import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.time_utils import ts_to_seconds, seconds_to_ts


def detect_shadow_clusters(
    mapped_events_df: pd.DataFrame,
    temporal_gap: int = 60,
) -> list:
    """Group co-occurring shadow events into temporal clusters.

    Scans events chronologically and groups consecutive shadow/deviation
    events that occur within ``temporal_gap`` seconds of each other.

    Parameters
    ----------
    mapped_events_df : pd.DataFrame
        Mapped event log with ``mapped_activity``, ``activity_name``,
        ``timestamp`` columns.
    temporal_gap : int
        Maximum gap in seconds between shadow events to belong to the
        same cluster.

    Returns
    -------
    list[dict]
        Each cluster dict contains:
        - ``cluster_id``: Integer identifier.
        - ``events``: List of event dicts within the cluster.
        - ``start_seconds``: Earliest event time in the cluster.
        - ``end_seconds``: Latest event time in the cluster.
        - ``duration``: Cluster span in seconds.
        - ``size``: Number of events.
        - ``activities``: List of distinct activity names.
        - ``preceding_formal``: The formal activity that occurred just
          before this cluster (or None).
        - ``following_formal``: The formal activity that occurred just
          after this cluster (or None).
    """
    if mapped_events_df is None or mapped_events_df.empty:
        return []

    if "mapped_activity" not in mapped_events_df.columns:
        return []

    df = mapped_events_df.copy()

    # Add seconds column for sorting
    if "raw_seconds" in df.columns:
        df["__secs"] = df["raw_seconds"].astype(int)
    elif "timestamp" in df.columns:
        df["__secs"] = df["timestamp"].apply(ts_to_seconds)
    else:
        return []

    df = df.sort_values("__secs").reset_index(drop=True)

    # Identify shadow vs formal
    df["__is_shadow"] = df["mapped_activity"].str.startswith("Deviation:", na=False)

    clusters = []
    current_cluster_events = []
    current_cluster_start = None

    for idx, row in df.iterrows():
        if not row["__is_shadow"]:
            # Formal event: close any open cluster
            if current_cluster_events:
                clusters.append(_finalize_cluster(
                    len(clusters),
                    current_cluster_events,
                    current_cluster_start,
                    df,
                    idx,
                ))
                current_cluster_events = []
                current_cluster_start = None
            continue

        # Shadow event
        event_secs = row["__secs"]

        if not current_cluster_events:
            # Start new cluster
            current_cluster_events.append(row.to_dict())
            current_cluster_start = event_secs
        elif (event_secs - current_cluster_events[-1]["__secs"]) <= temporal_gap:
            # Extend current cluster
            current_cluster_events.append(row.to_dict())
        else:
            # Gap too large: finalize current, start new
            clusters.append(_finalize_cluster(
                len(clusters),
                current_cluster_events,
                current_cluster_start,
                df,
                idx,
            ))
            current_cluster_events = [row.to_dict()]
            current_cluster_start = event_secs

    # Finalize any remaining cluster
    if current_cluster_events:
        clusters.append(_finalize_cluster(
            len(clusters),
            current_cluster_events,
            current_cluster_start,
            df,
            len(df),
        ))

    return clusters


def _finalize_cluster(cluster_id, events, start_secs, full_df, current_idx):
    """Build a cluster dict from accumulated shadow events."""
    end_secs = events[-1]["__secs"]
    activity_names = []
    for e in events:
        name = e.get("activity_name", e.get("mapped_activity", "Unknown"))
        activity_names.append(name)

    # Find preceding formal activity
    preceding_formal = None
    for i in range(current_idx - 1, -1, -1):
        if i < len(full_df) and not full_df.iloc[i].get("__is_shadow", True):
            preceding_formal = full_df.iloc[i].get(
                "mapped_activity",
                full_df.iloc[i].get("activity_name", None),
            )
            break

    # Find following formal activity
    following_formal = None
    for i in range(current_idx, len(full_df)):
        if not full_df.iloc[i].get("__is_shadow", True):
            following_formal = full_df.iloc[i].get(
                "mapped_activity",
                full_df.iloc[i].get("activity_name", None),
            )
            break

    # Clean up internal columns from event dicts
    clean_events = []
    for e in events:
        clean = {k: v for k, v in e.items() if not k.startswith("__")}
        clean_events.append(clean)

    return {
        "cluster_id": cluster_id,
        "events": clean_events,
        "start_seconds": start_secs,
        "end_seconds": end_secs,
        "start_timestamp": seconds_to_ts(start_secs),
        "end_timestamp": seconds_to_ts(end_secs),
        "duration": end_secs - start_secs,
        "size": len(events),
        "activities": list(dict.fromkeys(activity_names)),  # deduplicated, order preserved
        "preceding_formal": preceding_formal,
        "following_formal": following_formal,
    }


def classify_shadow_patterns(clusters: list) -> list:
    """Classify shadow clusters into structural patterns.

    Assigns a ``pattern_type`` to each cluster:

    - ``"isolated"``: Single event, no repetition elsewhere.
    - ``"sequential"``: Multiple distinct activities in temporal order.
    - ``"concurrent"``: Multiple events with overlapping/very short span
      relative to count (< 30s per event on average).
    - ``"recurring"``: Activity label appears in multiple clusters.

    Parameters
    ----------
    clusters : list[dict]
        Output from ``detect_shadow_clusters()``.

    Returns
    -------
    list[dict]
        Same clusters with additional keys:
        - ``pattern_type``: One of the types above.
        - ``pattern_confidence``: Float 0-1 indicating confidence.
        - ``pattern_description``: Human-readable explanation.
    """
    if not clusters:
        return []

    # Count how many clusters each activity label appears in
    activity_cluster_counts = Counter()
    for cluster in clusters:
        for act in cluster["activities"]:
            activity_cluster_counts[act] += 1

    classified = []
    for cluster in clusters:
        c = dict(cluster)  # shallow copy
        size = c["size"]
        duration = c["duration"]
        n_unique = len(c["activities"])

        if size == 1:
            # Single event: check if the activity recurs elsewhere
            act = c["activities"][0]
            if activity_cluster_counts.get(act, 0) > 1:
                c["pattern_type"] = "recurring"
                c["pattern_confidence"] = 0.7
                c["pattern_description"] = (
                    f"Single shadow event '{act}' that recurs in "
                    f"{activity_cluster_counts[act]} clusters total."
                )
            else:
                c["pattern_type"] = "isolated"
                c["pattern_confidence"] = 0.9
                c["pattern_description"] = (
                    f"Isolated one-off shadow event: '{act}'."
                )
        elif n_unique == 1:
            # Multiple events, same activity: recurring burst
            act = c["activities"][0]
            c["pattern_type"] = "recurring"
            c["pattern_confidence"] = 0.85
            c["pattern_description"] = (
                f"Burst of {size} repeated '{act}' events over "
                f"{duration}s."
            )
        elif duration > 0 and (duration / size) < 30:
            # Short average gap: likely concurrent
            c["pattern_type"] = "concurrent"
            c["pattern_confidence"] = 0.75
            c["pattern_description"] = (
                f"{size} shadow events ({n_unique} distinct activities) "
                f"in a {duration}s window, averaging "
                f"{duration / size:.0f}s apart (concurrent pattern)."
            )
        else:
            # Multiple distinct activities with longer gaps: sequential chain
            c["pattern_type"] = "sequential"
            c["pattern_confidence"] = 0.8
            c["pattern_description"] = (
                f"Sequential chain of {size} shadow events "
                f"({n_unique} distinct activities) spanning {duration}s."
            )

        # Enrich with context about surrounding formal activities
        context_parts = []
        if c.get("preceding_formal"):
            context_parts.append(f"after '{c['preceding_formal']}'")
        if c.get("following_formal"):
            context_parts.append(f"before '{c['following_formal']}'")
        if context_parts:
            c["pattern_description"] += (
                f" Occurs {' and '.join(context_parts)}."
            )

        classified.append(c)

    return classified


def get_pattern_summary(classified_clusters: list) -> dict:
    """Generate an aggregate summary of all shadow patterns.

    Parameters
    ----------
    classified_clusters : list[dict]
        Output from ``classify_shadow_patterns()``.

    Returns
    -------
    dict
        Summary with pattern type counts, total shadow events,
        most common shadow activities, and formal activity context.
    """
    if not classified_clusters:
        return {
            "total_clusters": 0,
            "total_shadow_events": 0,
            "pattern_counts": {},
            "top_shadow_activities": [],
            "formal_context_pairs": [],
        }

    pattern_counts = Counter()
    total_events = 0
    all_activities = Counter()
    context_pairs = []

    for cluster in classified_clusters:
        pattern_counts[cluster.get("pattern_type", "unknown")] += 1
        total_events += cluster["size"]
        for act in cluster["activities"]:
            all_activities[act] += 1
        if cluster.get("preceding_formal") and cluster.get("following_formal"):
            context_pairs.append((
                cluster["preceding_formal"],
                cluster["following_formal"],
                cluster.get("pattern_type", "unknown"),
            ))

    return {
        "total_clusters": len(classified_clusters),
        "total_shadow_events": total_events,
        "pattern_counts": dict(pattern_counts),
        "top_shadow_activities": all_activities.most_common(10),
        "formal_context_pairs": context_pairs,
        "avg_cluster_size": round(total_events / len(classified_clusters), 2),
        "avg_cluster_duration": round(
            sum(c["duration"] for c in classified_clusters)
            / len(classified_clusters),
            1,
        ),
    }


def shadow_patterns_to_dataframe(classified_clusters: list) -> pd.DataFrame:
    """Convert classified clusters to a flat DataFrame for export/analysis.

    Parameters
    ----------
    classified_clusters : list[dict]
        Output from ``classify_shadow_patterns()``.

    Returns
    -------
    pd.DataFrame
        One row per cluster with columns: cluster_id, pattern_type,
        pattern_confidence, size, duration, start_timestamp,
        end_timestamp, activities, preceding_formal, following_formal,
        pattern_description.
    """
    if not classified_clusters:
        return pd.DataFrame()

    rows = []
    for c in classified_clusters:
        rows.append({
            "cluster_id": c["cluster_id"],
            "pattern_type": c.get("pattern_type", "unknown"),
            "pattern_confidence": c.get("pattern_confidence", 0.0),
            "size": c["size"],
            "duration": c["duration"],
            "start_timestamp": c.get("start_timestamp", ""),
            "end_timestamp": c.get("end_timestamp", ""),
            "activities": "; ".join(c["activities"]),
            "preceding_formal": c.get("preceding_formal", ""),
            "following_formal": c.get("following_formal", ""),
            "pattern_description": c.get("pattern_description", ""),
        })

    return pd.DataFrame(rows)
