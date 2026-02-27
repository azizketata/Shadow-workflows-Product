#!/usr/bin/env python3
"""CLI runner for POWL shadow workflow discovery.

Usage:
    python research/run_powl_analysis.py \
        --mapped-events results/mapped_events.csv \
        --output-dir results/powl
"""

import sys
import os
import argparse
import json
from unittest.mock import MagicMock

# Mock Streamlit before importing project modules
sys.modules["streamlit"] = MagicMock()

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.powl.powl_discovery import (
    discover_shadow_powl,
    discover_full_powl,
    visualize_powl,
)
from research.powl.powl_analysis import get_powl_statistics, summarize_powl_model
from research.powl.shadow_patterns import (
    detect_shadow_clusters,
    classify_shadow_patterns,
    get_pattern_summary,
    shadow_patterns_to_dataframe,
)


def main():
    parser = argparse.ArgumentParser(
        description="POWL discovery for shadow workflow analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mapped-events", required=True,
                        help="Path to mapped_events.csv")
    parser.add_argument("--output-dir", default="results/powl",
                        help="Directory for output files")
    parser.add_argument("--frequency-threshold", type=float, default=0.0,
                        help="POWL discovery frequency threshold")
    parser.add_argument("--temporal-gap", type=int, default=60,
                        help="Temporal gap (seconds) for shadow clustering")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  POWL Shadow Workflow Analysis")
    print("=" * 60)

    # Load data
    df = pd.read_csv(args.mapped_events)
    print(f"\nLoaded {len(df)} mapped events from {args.mapped_events}")

    if "mapped_activity" not in df.columns:
        print("ERROR: mapped_events.csv must have 'mapped_activity' column.")
        sys.exit(1)

    shadow_count = df["mapped_activity"].str.startswith("Deviation:", na=False).sum()
    formal_count = len(df) - shadow_count
    print(f"  Formal events: {formal_count}")
    print(f"  Shadow events: {shadow_count}")

    # --- Shadow POWL Discovery ---
    print("\n--- Shadow POWL Discovery ---")
    shadow_powl, n_shadow = discover_shadow_powl(df, args.frequency_threshold)
    if shadow_powl is not None:
        stats = get_powl_statistics(shadow_powl)
        print(f"  Shadow POWL: {stats.get('node_count', '?')} nodes, "
              f"{stats.get('unique_activities', '?')} activities")
        print(summarize_powl_model(shadow_powl))

        try:
            out_path = os.path.join(args.output_dir, "shadow_powl.svg")
            visualize_powl(shadow_powl, out_path)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  Visualization skipped: {e}")

        with open(os.path.join(args.output_dir, "shadow_powl_stats.json"), "w") as f:
            json.dump(stats, f, indent=2, default=str)
    else:
        print("  Not enough shadow events for POWL discovery.")

    # --- Full POWL Discovery ---
    print("\n--- Full POWL Discovery ---")
    full_powl, n_full = discover_full_powl(df, args.frequency_threshold)
    if full_powl is not None:
        stats = get_powl_statistics(full_powl)
        print(f"  Full POWL: {stats.get('node_count', '?')} nodes, "
              f"{stats.get('unique_activities', '?')} activities")

        try:
            out_path = os.path.join(args.output_dir, "full_powl.svg")
            visualize_powl(full_powl, out_path)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  Visualization skipped: {e}")
    else:
        print("  Not enough events for full POWL discovery.")

    # --- Shadow Pattern Analysis ---
    print("\n--- Shadow Pattern Clustering ---")
    clusters = detect_shadow_clusters(df, temporal_gap=args.temporal_gap)
    if clusters:
        classified = classify_shadow_patterns(clusters)
        summary = get_pattern_summary(classified)
        print(f"  Found {summary['total_clusters']} shadow clusters")
        for ptype, count in summary.get("pattern_distribution", {}).items():
            print(f"    {ptype}: {count}")

        patterns_df = shadow_patterns_to_dataframe(classified)
        out_csv = os.path.join(args.output_dir, "shadow_patterns.csv")
        patterns_df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

        with open(os.path.join(args.output_dir, "shadow_patterns.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
    else:
        print("  No shadow clusters found.")

    print(f"\n  All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
