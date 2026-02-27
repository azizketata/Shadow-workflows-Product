#!/usr/bin/env python3
"""CLI runner for longitudinal / multi-meeting analysis.

Usage:
    python research/run_longitudinal.py \
        --meeting-results results/meeting1.json results/meeting2.json ... \
        --output-dir results/longitudinal

Each meeting JSON should have:
{
    "name": "Council Meeting Jan 2026",
    "date": "2026-01-20",
    "fitness_dedup": 0.654,
    "fitness_raw": 0.450,
    "agenda_coverage": 100.0,
    "shadow_count": 14,
    "event_count": 776,
    "shadows": ["Shadow: Extended Discussion", ...]
}
"""

import sys
import os
import argparse
import json
from unittest.mock import MagicMock

sys.modules["streamlit"] = MagicMock()

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.longitudinal.drift_detector import (
    compute_fitness_trend,
    track_shadow_evolution,
)
from research.longitudinal.meeting_comparator import (
    compare_meetings,
    generate_comparison_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Longitudinal multi-meeting analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--meeting-results", nargs="+", required=True,
                        help="Paths to meeting result JSON files")
    parser.add_argument("--output-dir", default="results/longitudinal",
                        help="Directory for output files")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Longitudinal Multi-Meeting Analysis")
    print("=" * 60)

    # Load meeting results
    meetings = []
    for path in args.meeting_results:
        with open(path, "r", encoding="utf-8") as f:
            meetings.append(json.load(f))
    print(f"\nLoaded {len(meetings)} meeting results")

    for m in meetings:
        print(f"  - {m.get('name', 'Unknown')}: fitness={m.get('fitness_dedup', 0):.1%}, "
              f"shadows={m.get('shadow_count', 0)}")

    if len(meetings) < 2:
        print("\nNeed at least 2 meetings for longitudinal analysis.")
        print("Run the pipeline on multiple meeting videos first.")
        sys.exit(0)

    # Fitness trend
    print("\n--- Fitness Trend Analysis ---")
    trend = compute_fitness_trend(meetings)
    print(f"  Trend: {trend.get('trend_direction', 'unknown')}")
    print(f"  Slope: {trend.get('slope', 0):.4f} per meeting")
    if trend.get("change_points"):
        print(f"  Change points: {trend['change_points']}")

    # Shadow evolution
    print("\n--- Shadow Workflow Evolution ---")
    shadow_data = [
        {"meeting_name": m.get("name", f"Meeting {i}"),
         "shadows": m.get("shadows", [])}
        for i, m in enumerate(meetings)
    ]
    evolution = track_shadow_evolution(shadow_data)
    if evolution.get("persistent_shadows"):
        print(f"  Persistent shadows (>50% of meetings): {evolution['persistent_shadows']}")
    if evolution.get("emerging_shadows"):
        print(f"  Emerging shadows: {evolution['emerging_shadows']}")
    if evolution.get("one_off_shadows"):
        print(f"  One-off shadows: {len(evolution['one_off_shadows'])}")

    # Pairwise comparison (last two meetings)
    if len(meetings) >= 2:
        print("\n--- Pairwise Comparison (last two meetings) ---")
        comparison = compare_meetings(meetings[-2], meetings[-1])
        print(f"  Structural similarity: {comparison.get('structural_similarity', 0):.3f}")
        print(f"  Activity Jaccard: {comparison.get('activity_jaccard', 0):.3f}")
        print(f"  Fitness difference: {comparison.get('fitness_diff', 0):+.1%}")

        report = generate_comparison_report(comparison)
        report_path = os.path.join(args.output_dir, "comparison_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Saved: {report_path}")

    # Save all results
    results = {
        "meetings_analyzed": len(meetings),
        "fitness_trend": trend,
        "shadow_evolution": evolution,
    }
    results_path = os.path.join(args.output_dir, "longitudinal_analysis.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
