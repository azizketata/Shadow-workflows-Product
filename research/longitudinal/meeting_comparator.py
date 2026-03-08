"""Compare two meeting analysis results structurally.

Provides pairwise comparison of meetings based on their activity sets,
shadow workflows, fitness scores, and structural similarity. Useful for
identifying how specific meetings differ and what changed between sessions.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute the Jaccard similarity coefficient between two sets.

    J(A, B) = |A intersection B| / |A union B|

    Args:
        set_a: First set of elements.
        set_b: Second set of elements.

    Returns:
        Float between 0.0 (completely disjoint) and 1.0 (identical).
        Returns 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b

    if not union:
        return 0.0

    return len(intersection) / len(union)


def compare_meetings(meeting_a: dict, meeting_b: dict) -> dict:
    """Compare two meeting analysis results structurally.

    Performs a multi-dimensional comparison including activity overlap,
    shadow workflow similarity, fitness differences, and sequence
    similarity.

    Args:
        meeting_a: First meeting dict with keys:
            - mapped_events (pd.DataFrame): Event log with 'mapped_activity'
                or 'activity_name' column
            - fitness_dedup (float): Deduplicated fitness score
            - shadow_count (int): Number of shadow events
            - agenda_coverage (float): Percentage of agenda items matched
            - shadows (list[str]): List of shadow workflow labels
            - name (str, optional): Meeting name
        meeting_b: Second meeting dict with same structure.

    Returns:
        Dictionary with:
        - shared_activities: List of activities present in both meetings
        - unique_to_a: List of activities only in meeting A
        - unique_to_b: List of activities only in meeting B
        - activity_jaccard: Float Jaccard similarity of activity sets
        - fitness_diff: Float (B.fitness - A.fitness)
        - fitness_a: Float
        - fitness_b: Float
        - shadow_overlap: List of shadow labels present in both
        - shadow_jaccard: Float Jaccard similarity of shadow label sets
        - shadows_unique_to_a: List of shadows only in A
        - shadows_unique_to_b: List of shadows only in B
        - agenda_coverage_diff: Float (B.coverage - A.coverage)
        - structural_similarity: Float (0-1) composite similarity score
        - sequence_similarity: Float (0-1) based on longest common subsequence
        - detail: Dict with additional breakdown information
    """
    # Extract activity sets
    activities_a = _extract_activity_set(meeting_a)
    activities_b = _extract_activity_set(meeting_b)

    shared_activities = sorted(activities_a & activities_b)
    unique_to_a = sorted(activities_a - activities_b)
    unique_to_b = sorted(activities_b - activities_a)
    activity_jaccard = compute_jaccard_similarity(activities_a, activities_b)

    # Fitness comparison
    fitness_a = float(meeting_a.get("fitness_dedup", 0.0))
    fitness_b = float(meeting_b.get("fitness_dedup", 0.0))
    fitness_diff = fitness_b - fitness_a

    # Shadow comparison
    shadows_a = set(meeting_a.get("shadows", []))
    shadows_b = set(meeting_b.get("shadows", []))
    shadow_overlap = sorted(shadows_a & shadows_b)
    shadows_unique_to_a = sorted(shadows_a - shadows_b)
    shadows_unique_to_b = sorted(shadows_b - shadows_a)
    shadow_jaccard = compute_jaccard_similarity(shadows_a, shadows_b)

    # Agenda coverage comparison
    coverage_a = float(meeting_a.get("agenda_coverage", 0.0))
    coverage_b = float(meeting_b.get("agenda_coverage", 0.0))
    agenda_coverage_diff = coverage_b - coverage_a

    # Sequence similarity (LCS-based)
    seq_a = _extract_activity_sequence(meeting_a)
    seq_b = _extract_activity_sequence(meeting_b)
    sequence_similarity = _normalized_lcs_similarity(seq_a, seq_b)

    # Activity frequency comparison
    freq_a = _extract_activity_frequencies(meeting_a)
    freq_b = _extract_activity_frequencies(meeting_b)
    frequency_similarity = _cosine_similarity_counters(freq_a, freq_b)

    # Composite structural similarity
    # Weighted average: activity overlap (40%), sequence (30%), frequency (30%)
    structural_similarity = (
        0.4 * activity_jaccard
        + 0.3 * sequence_similarity
        + 0.3 * frequency_similarity
    )

    # Shadow count comparison
    shadow_count_a = int(meeting_a.get("shadow_count", len(shadows_a)))
    shadow_count_b = int(meeting_b.get("shadow_count", len(shadows_b)))

    return {
        "shared_activities": shared_activities,
        "unique_to_a": unique_to_a,
        "unique_to_b": unique_to_b,
        "activity_jaccard": round(activity_jaccard, 4),
        "fitness_diff": round(fitness_diff, 2),
        "fitness_a": round(fitness_a, 2),
        "fitness_b": round(fitness_b, 2),
        "shadow_overlap": shadow_overlap,
        "shadow_jaccard": round(shadow_jaccard, 4),
        "shadows_unique_to_a": shadows_unique_to_a,
        "shadows_unique_to_b": shadows_unique_to_b,
        "agenda_coverage_diff": round(agenda_coverage_diff, 2),
        "structural_similarity": round(structural_similarity, 4),
        "sequence_similarity": round(sequence_similarity, 4),
        "detail": {
            "name_a": meeting_a.get("name", "Meeting A"),
            "name_b": meeting_b.get("name", "Meeting B"),
            "total_activities_a": len(activities_a),
            "total_activities_b": len(activities_b),
            "shadow_count_a": shadow_count_a,
            "shadow_count_b": shadow_count_b,
            "agenda_coverage_a": round(coverage_a, 2),
            "agenda_coverage_b": round(coverage_b, 2),
            "frequency_similarity": round(frequency_similarity, 4),
            "sequence_length_a": len(seq_a),
            "sequence_length_b": len(seq_b),
        },
    }


def generate_comparison_report(comparison: dict) -> str:
    """Generate a human-readable Markdown comparison report.

    Args:
        comparison: Dict returned by compare_meetings().

    Returns:
        Markdown-formatted comparison report string.
    """
    detail = comparison.get("detail", {})
    name_a = detail.get("name_a", "Meeting A")
    name_b = detail.get("name_b", "Meeting B")

    lines: list[str] = []
    lines.append(f"# Meeting Comparison: {name_a} vs {name_b}\n")

    # Overview
    lines.append("## Overview\n")
    lines.append(f"| Metric | {name_a} | {name_b} | Difference |")
    lines.append("|--------|------:|------:|----------:|")
    lines.append(
        f"| Fitness (dedup) | {comparison['fitness_a']}% | "
        f"{comparison['fitness_b']}% | "
        f"{comparison['fitness_diff']:+.2f}% |"
    )
    lines.append(
        f"| Agenda Coverage | {detail.get('agenda_coverage_a', 0)}% | "
        f"{detail.get('agenda_coverage_b', 0)}% | "
        f"{comparison['agenda_coverage_diff']:+.2f}% |"
    )
    lines.append(
        f"| Shadow Count | {detail.get('shadow_count_a', 0)} | "
        f"{detail.get('shadow_count_b', 0)} | "
        f"{detail.get('shadow_count_b', 0) - detail.get('shadow_count_a', 0):+d} |"
    )
    lines.append(
        f"| Unique Activities | {detail.get('total_activities_a', 0)} | "
        f"{detail.get('total_activities_b', 0)} | -- |"
    )
    lines.append("")

    # Similarity scores
    lines.append("## Similarity Scores\n")
    lines.append(f"- **Structural similarity:** {comparison['structural_similarity']:.2%}")
    lines.append(f"- **Activity Jaccard:** {comparison['activity_jaccard']:.2%}")
    lines.append(f"- **Sequence similarity:** {comparison['sequence_similarity']:.2%}")
    lines.append(
        f"- **Frequency similarity:** {detail.get('frequency_similarity', 0):.2%}"
    )
    lines.append(f"- **Shadow Jaccard:** {comparison['shadow_jaccard']:.2%}\n")

    # Activity overlap
    lines.append("## Activity Overlap\n")
    shared = comparison.get("shared_activities", [])
    if shared:
        lines.append(f"**Shared ({len(shared)}):** {', '.join(shared)}\n")
    unique_a = comparison.get("unique_to_a", [])
    if unique_a:
        lines.append(f"**Only in {name_a} ({len(unique_a)}):** {', '.join(unique_a)}\n")
    unique_b = comparison.get("unique_to_b", [])
    if unique_b:
        lines.append(f"**Only in {name_b} ({len(unique_b)}):** {', '.join(unique_b)}\n")

    # Shadow workflow comparison
    lines.append("## Shadow Workflow Comparison\n")
    shadow_overlap = comparison.get("shadow_overlap", [])
    if shadow_overlap:
        lines.append(
            f"**Recurring shadows ({len(shadow_overlap)}):** {', '.join(shadow_overlap)}\n"
        )
    su_a = comparison.get("shadows_unique_to_a", [])
    if su_a:
        lines.append(f"**Only in {name_a}:** {', '.join(su_a)}\n")
    su_b = comparison.get("shadows_unique_to_b", [])
    if su_b:
        lines.append(f"**Only in {name_b}:** {', '.join(su_b)}\n")

    if not shadow_overlap and not su_a and not su_b:
        lines.append("No shadow workflows recorded for comparison.\n")

    # Interpretation
    lines.append("## Interpretation\n")
    sim = comparison["structural_similarity"]
    if sim > 0.8:
        lines.append(
            "These meetings are **highly similar** in structure, suggesting "
            "a stable and repeatable meeting process."
        )
    elif sim > 0.5:
        lines.append(
            "These meetings share **moderate structural similarity**. "
            "Core procedures are consistent, with some variation in "
            "specific agenda items or shadow workflows."
        )
    elif sim > 0.2:
        lines.append(
            "These meetings have **low structural similarity**, indicating "
            "significant process differences. This could reflect different "
            "agenda types, procedural changes, or concept drift."
        )
    else:
        lines.append(
            "These meetings are **structurally dissimilar**, suggesting "
            "fundamentally different meeting types or major process changes."
        )

    lines.append("")
    lines.append("---\n")
    lines.append(
        "*Generated by Meeting Process Twin -- Research/Longitudinal Module*"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_activity_set(meeting: dict) -> set[str]:
    """Extract the unique set of activity names from a meeting dict."""
    activities: set[str] = set()

    df = meeting.get("mapped_events")
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        # Try different column names in priority order
        for col in ["mapped_activity", "concept:name", "activity_name"]:
            if col in df.columns:
                values = df[col].dropna().astype(str).unique()
                activities.update(v for v in values if v and v != "nan")
                break

    return activities


def _extract_activity_sequence(meeting: dict) -> list[str]:
    """Extract the ordered sequence of activities from a meeting dict."""
    df = meeting.get("mapped_events")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    # Find the activity column
    act_col = None
    for col in ["mapped_activity", "concept:name", "activity_name"]:
        if col in df.columns:
            act_col = col
            break

    if act_col is None:
        return []

    return df[act_col].dropna().astype(str).tolist()


def _extract_activity_frequencies(meeting: dict) -> Counter:
    """Extract activity frequency counts from a meeting dict."""
    seq = _extract_activity_sequence(meeting)
    return Counter(seq)


def _normalized_lcs_similarity(seq_a: list[str], seq_b: list[str]) -> float:
    """Compute normalized Longest Common Subsequence similarity.

    Returns a value between 0.0 (no common subsequence) and 1.0 (identical).
    Normalized by the length of the longer sequence.

    Uses an optimized O(n*m) dynamic programming approach with space
    optimization (two rows instead of full matrix).

    Args:
        seq_a: First sequence of activity labels.
        seq_b: Second sequence of activity labels.

    Returns:
        Float similarity score between 0.0 and 1.0.
    """
    n = len(seq_a)
    m = len(seq_b)

    if n == 0 or m == 0:
        return 0.0

    # Limit sequence lengths to avoid excessive computation
    max_len = 500
    if n > max_len:
        # Sample evenly
        step = n // max_len
        seq_a = seq_a[::step][:max_len]
        n = len(seq_a)
    if m > max_len:
        step = m // max_len
        seq_b = seq_b[::step][:max_len]
        m = len(seq_b)

    # Space-optimized LCS: two rows
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (m + 1)

    lcs_length = prev[m]
    max_length = max(n, m)

    return lcs_length / max_length if max_length > 0 else 0.0


def _cosine_similarity_counters(counter_a: Counter, counter_b: Counter) -> float:
    """Compute cosine similarity between two frequency Counters.

    Treats each Counter as a sparse vector over the union of all keys.

    Args:
        counter_a: First frequency counter.
        counter_b: Second frequency counter.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not counter_a or not counter_b:
        return 0.0

    all_keys = set(counter_a.keys()) | set(counter_b.keys())

    dot_product = sum(
        counter_a.get(k, 0) * counter_b.get(k, 0) for k in all_keys
    )
    magnitude_a = sum(v ** 2 for v in counter_a.values()) ** 0.5
    magnitude_b = sum(v ** 2 for v in counter_b.values()) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)
