#!/usr/bin/env python3
"""CLI runner for process deviance classification.

Usage:
    python research/run_deviance_analysis.py \
        --mapped-events results/mapped_events.csv \
        --output-dir results/deviance
"""

import sys
import os
import argparse
import json
from unittest.mock import MagicMock

sys.modules["streamlit"] = MagicMock()

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.deviance.deviance_classifier import DevianceClassifier
from research.deviance.deviance_report import generate_deviance_report


def main():
    parser = argparse.ArgumentParser(
        description="Process deviance classification for shadow workflows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mapped-events", required=True,
                        help="Path to mapped_events.csv")
    parser.add_argument("--output-dir", default="results/deviance",
                        help="Directory for output files")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (optional, for LLM classification)")
    parser.add_argument("--declare-results", default=None,
                        help="Path to Roberts Rules conformance JSON (optional)")
    parser.add_argument("--meeting-name", default="Council Meeting")
    parser.add_argument("--meeting-date", default="")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Process Deviance Classification")
    print("=" * 60)

    # Load data
    df = pd.read_csv(args.mapped_events)
    print(f"\nLoaded {len(df)} mapped events from {args.mapped_events}")

    shadow_mask = df.get("mapped_activity", pd.Series()).str.startswith("Deviation:", na=False)
    print(f"  Shadow events: {shadow_mask.sum()}")

    # Load declare violations (optional)
    declare_violations = None
    if args.declare_results and os.path.exists(args.declare_results):
        with open(args.declare_results, "r") as f:
            declare_data = json.load(f)
        declare_violations = declare_data.get("violations", [])
        print(f"  Loaded {len(declare_violations)} Declare violations for cross-reference")

    # Classify
    print("\n--- Classifying Shadow Workflows ---")
    classifier = DevianceClassifier(api_key=args.api_key)
    classified = classifier.classify(df, declare_violations=declare_violations)

    # Summary
    summary = classifier.generate_deviance_summary(classified)
    print(f"\n  Category distribution:")
    for cat, count in summary.get("category_distribution", {}).items():
        print(f"    {cat}: {count}")
    print(f"  Severity score: {summary.get('severity_score', 0)}")

    # Save classified events
    out_csv = os.path.join(args.output_dir, "classified_events.csv")
    classified.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "deviance_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")

    # Generate report
    report = generate_deviance_report(classified, summary, args.meeting_name, args.meeting_date)
    report_path = os.path.join(args.output_dir, "deviance_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    print(f"\n  All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
