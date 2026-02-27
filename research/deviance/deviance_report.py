"""Generate thesis-quality deviance analysis reports in Markdown.

Produces structured reports suitable for academic papers or governance
reviews, including statistical breakdowns, severity analysis, and
actionable recommendations.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Optional

import pandas as pd

from research.deviance.deviance_taxonomy import (
    CATEGORY_DESCRIPTIONS,
    DevianceCategory,
    get_severity,
)


def generate_deviance_report(
    classified_df: pd.DataFrame,
    summary: dict,
    meeting_name: str = "Meeting",
    meeting_date: str | None = None,
) -> str:
    """Generate a Markdown report analyzing process deviance patterns.

    Creates a multi-section report with:
    - Executive summary
    - Category breakdown with percentages and severity
    - Timeline analysis of deviance events
    - Procedural violation details (if any)
    - Recommendations
    - Methodology notes

    Args:
        classified_df: DataFrame with 'deviance_category' and
            'deviance_rationale' columns, as returned by
            DevianceClassifier.classify().
        summary: Summary dict from DevianceClassifier.generate_deviance_summary().
        meeting_name: Human-readable name for the meeting.
        meeting_date: Optional date string (YYYY-MM-DD). Defaults to today.

    Returns:
        Complete Markdown report as a string.
    """
    if meeting_date is None:
        meeting_date = datetime.now().strftime("%Y-%m-%d")

    sections: list[str] = []

    # ---- Title ----
    sections.append(f"# Process Deviance Analysis: {meeting_name}")
    sections.append(f"**Date:** {meeting_date}  ")
    sections.append(
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    # ---- Executive Summary ----
    sections.append("## Executive Summary\n")
    total_shadow = summary.get("total_shadow_events", 0)
    total_formal = summary.get("total_formal_events", 0)
    total_events = total_shadow + total_formal
    severity_pct = summary.get("severity_percentage", 0.0)

    if total_events > 0:
        shadow_pct = total_shadow / total_events * 100
    else:
        shadow_pct = 0.0

    sections.append(
        f"This meeting contained **{total_events}** process events, of which "
        f"**{total_shadow}** ({shadow_pct:.1f}%) were identified as shadow "
        f"workflow events (deviations from the formal agenda). "
        f"The overall severity score is **{severity_pct:.1f}%** "
        f"(0% = entirely benign, 100% = all procedural violations).\n"
    )

    # Quick verdict
    if severity_pct < 10:
        verdict = "LOW -- Meeting followed formal procedures with minimal deviance."
    elif severity_pct < 30:
        verdict = "MODERATE -- Some procedural deviations detected; review recommended."
    elif severity_pct < 60:
        verdict = "ELEVATED -- Significant procedural concerns requiring attention."
    else:
        verdict = "HIGH -- Substantial procedural violations detected; governance review needed."

    sections.append(f"**Deviance Risk Level:** {verdict}\n")

    # ---- Category Breakdown ----
    sections.append("## Deviance Category Breakdown\n")
    cat_dist = summary.get("category_distribution", {})

    if cat_dist:
        sections.append(
            "| Category | Count | % of Shadow | Severity Weight | Description |"
        )
        sections.append(
            "|----------|------:|------------:|----------------:|-------------|"
        )
        for cat in DevianceCategory:
            count = cat_dist.get(cat.value, 0)
            if count == 0:
                continue
            pct = count / total_shadow * 100 if total_shadow > 0 else 0
            weight = get_severity(cat)
            desc = CATEGORY_DESCRIPTIONS.get(cat, "")
            # Truncate description for table
            short_desc = desc[:80] + "..." if len(desc) > 80 else desc
            sections.append(
                f"| {cat.name} | {count} | {pct:.1f}% | {weight} | {short_desc} |"
            )
        sections.append("")
    else:
        sections.append("No deviance categories recorded.\n")

    # ---- Timeline Analysis ----
    sections.append("## Timeline Analysis\n")

    if classified_df is not None and not classified_df.empty:
        shadow_mask = classified_df["deviance_category"] != "N/A"
        shadow_df = classified_df[shadow_mask].copy()

        if not shadow_df.empty and "timestamp" in shadow_df.columns:
            sections.append(
                "Shadow events ordered by time of occurrence:\n"
            )
            sections.append(
                "| Time | Activity | Category | Rationale |"
            )
            sections.append(
                "|------|----------|----------|-----------|"
            )
            for _, row in shadow_df.iterrows():
                ts = row.get("timestamp", "??:??:??")
                act = str(row.get("activity_name", ""))[:60]
                cat = row.get("deviance_category", "unknown")
                rat = row.get("deviance_rationale", "")
                sections.append(f"| {ts} | {act} | {cat} | {rat} |")

            sections.append("")

            # Temporal clustering: identify periods of high deviance
            sections.append("### Temporal Clustering\n")
            temporal_summary = _analyze_temporal_clusters(shadow_df)
            if temporal_summary:
                sections.append(temporal_summary)
            else:
                sections.append(
                    "Insufficient timestamp data for temporal clustering.\n"
                )
        else:
            sections.append("No timestamp data available for timeline analysis.\n")
    else:
        sections.append("No classified events available for timeline analysis.\n")

    # ---- Procedural Violations Detail ----
    violations = summary.get("top_violations", [])
    if violations:
        sections.append("## Procedural Violations (Detail)\n")
        sections.append(
            "The following events were classified as procedural violations "
            "and may affect the legitimacy of meeting decisions:\n"
        )
        for i, v in enumerate(violations, 1):
            sections.append(f"### Violation {i}")
            sections.append(f"- **Time:** {v.get('timestamp', 'N/A')}")
            sections.append(f"- **Activity:** {v.get('activity_name', 'N/A')}")
            sections.append(f"- **Details:** {v.get('details', 'N/A')}")
            sections.append(
                f"- **Classification method:** {v.get('rationale', 'N/A')}\n"
            )

    # ---- Classification Method ----
    method_dist = summary.get("classification_method", {})
    if method_dist:
        sections.append("## Classification Methodology\n")
        sections.append("| Method | Count |")
        sections.append("|--------|------:|")
        for method, count in sorted(method_dist.items(), key=lambda x: -x[1]):
            sections.append(f"| {method} | {count} |")
        sections.append("")

        rule_count = sum(
            c for m, c in method_dist.items() if m.startswith("rule:")
        )
        llm_count = method_dist.get("llm", 0)
        unknown_count = sum(
            c for m, c in method_dist.items() if m.startswith("unknown:")
        )
        sections.append(
            f"Rule-based classifications: **{rule_count}** | "
            f"LLM-based: **{llm_count}** | "
            f"Unresolved: **{unknown_count}**\n"
        )

    # ---- Recommendations ----
    recommendations = summary.get("recommendations", [])
    if recommendations:
        sections.append("## Recommendations\n")
        for i, rec in enumerate(recommendations, 1):
            sections.append(f"{i}. {rec}")
        sections.append("")

    # ---- Methodology Notes ----
    sections.append("## Methodology Notes\n")
    sections.append(
        "This report was generated by the Meeting Process Twin deviance "
        "analysis module. Classification uses a two-phase approach:\n"
    )
    sections.append(
        "1. **Rule-based classification**: Deterministic keyword pattern matching "
        "against a taxonomy of known deviance types in civic meetings.\n"
        "2. **LLM fallback** (optional): Events not matched by rules are sent "
        "to a language model (GPT-4o-mini) for classification using a structured "
        "prompt with category definitions.\n"
    )
    sections.append(
        "Severity scores are weighted: Procedural Violations (3), "
        "External Disruptions (2), Unknown (1), Benign/Efficiency/Innovation (0). "
        "The percentage is computed relative to the maximum possible severity "
        "(all events being procedural violations).\n"
    )

    sections.append("---\n")
    sections.append(
        "*Generated by Meeting Process Twin -- Research/Deviance Module*"
    )

    return "\n".join(sections)


def _analyze_temporal_clusters(shadow_df: pd.DataFrame) -> str | None:
    """Analyze temporal clusters of shadow events.

    Groups shadow events into 5-minute bins and identifies periods
    of high deviance density.

    Args:
        shadow_df: Shadow-only DataFrame with 'timestamp' column.

    Returns:
        Markdown string describing temporal clusters, or None.
    """
    if "timestamp" not in shadow_df.columns:
        return None

    # Convert timestamps to seconds for binning
    from pipeline.time_utils import ts_to_seconds

    seconds = shadow_df["timestamp"].apply(ts_to_seconds)
    if seconds.max() == 0 and len(seconds) > 1:
        return None

    # 5-minute bins
    bin_size = 300  # 5 minutes
    bins = (seconds // bin_size) * bin_size
    bin_counts = bins.value_counts().sort_index()

    if bin_counts.empty:
        return None

    lines: list[str] = []
    lines.append("Deviance density by 5-minute window:\n")
    lines.append("| Window | Shadow Events | Density |")
    lines.append("|--------|-------------:|---------|")

    max_count = bin_counts.max()
    for bin_start, count in bin_counts.items():
        from pipeline.time_utils import seconds_to_ts

        start_str = seconds_to_ts(int(bin_start))
        end_str = seconds_to_ts(int(bin_start) + bin_size)
        bar = "#" * int(count / max_count * 10) if max_count > 0 else ""
        lines.append(f"| {start_str}-{end_str} | {count} | {bar} |")

    # Identify peak deviance windows (top 3)
    top_bins = bin_counts.nlargest(3)
    if len(top_bins) > 0:
        lines.append("")
        lines.append("**Peak deviance windows:**")
        for bin_start, count in top_bins.items():
            from pipeline.time_utils import seconds_to_ts as _sts

            start_str = _sts(int(bin_start))
            end_str = _sts(int(bin_start) + bin_size)
            # Get categories in this window
            window_mask = bins == bin_start
            cats = shadow_df.loc[window_mask, "deviance_category"].value_counts()
            cat_str = ", ".join(f"{c}({n})" for c, n in cats.items())
            lines.append(f"- {start_str}-{end_str}: {count} events [{cat_str}]")

    lines.append("")
    return "\n".join(lines)
