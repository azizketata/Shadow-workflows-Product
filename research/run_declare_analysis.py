#!/usr/bin/env python3
"""CLI runner for Declare constraint analysis (Robert's Rules).

Usage:
    python research/run_declare_analysis.py \
        --mapped-events results/mapped_events.csv \
        --agenda agenda.txt \
        --output-dir results/declare
"""

import sys
import os
import argparse
import json
from unittest.mock import MagicMock

sys.modules["streamlit"] = MagicMock()

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.declare.roberts_rules import (
    ROBERTS_RULES_CONSTRAINTS,
    get_constraints_for_agenda,
)
from research.declare.declare_conformance import (
    check_roberts_rules_conformance,
    compute_procedural_compliance_score,
)
from research.declare.violation_analysis import (
    classify_violations,
    get_violation_summary,
    generate_procedural_report,
)


def load_agenda_activities(path: str) -> list:
    """Extract activity names from agenda file."""
    activities = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip numbering (e.g., "1. Call to Order" → "Call to Order")
            import re
            cleaned = re.sub(r"^\d+[a-z]?\.\s*", "", line)
            if cleaned and cleaned != line.split("\n")[0].strip():
                activities.append(cleaned)
            elif cleaned:
                activities.append(cleaned)
    return [a for a in activities if len(a) > 2]


def main():
    parser = argparse.ArgumentParser(
        description="Declare constraint analysis — Robert's Rules of Order",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mapped-events", required=True,
                        help="Path to mapped_events.csv")
    parser.add_argument("--agenda", required=True,
                        help="Path to agenda text file")
    parser.add_argument("--output-dir", default="results/declare",
                        help="Directory for output files")
    parser.add_argument("--meeting-name", default="Council Meeting",
                        help="Meeting name for the report")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Declare Constraint Analysis — Robert's Rules")
    print("=" * 60)

    # Load data
    df = pd.read_csv(args.mapped_events)
    print(f"\nLoaded {len(df)} mapped events from {args.mapped_events}")

    agenda = load_agenda_activities(args.agenda)
    print(f"Loaded {len(agenda)} agenda activities from {args.agenda}")

    # Filter applicable constraints
    applicable = get_constraints_for_agenda(agenda)
    print(f"\nRobert's Rules constraints: {len(ROBERTS_RULES_CONSTRAINTS)} total, "
          f"{len(applicable)} applicable to this agenda")

    # Run conformance check
    print("\n--- Conformance Checking ---")
    results = check_roberts_rules_conformance(df, agenda)

    satisfied = sum(1 for r in results if r.get("satisfied"))
    violated = len(results) - satisfied
    print(f"  Satisfied: {satisfied}/{len(results)}")
    print(f"  Violated:  {violated}/{len(results)}")

    for r in results:
        status = "PASS" if r["satisfied"] else "FAIL"
        print(f"    [{status}] {r['template']}: {r['activity_a']} -> {r.get('activity_b', 'N/A')}")

    # Compute score
    score_result = compute_procedural_compliance_score(results)
    print(f"\n  Procedural Compliance Score: {score_result['score']:.1f}/100 "
          f"(Grade: {score_result['grade']})")

    # Classify violations
    violations = classify_violations(results)
    if violations:
        summary = get_violation_summary(violations)
        print(f"\n--- Violation Summary ---")
        print(f"  By severity: {summary.get('by_severity', {})}")
    else:
        summary = {"total": 0, "by_severity": {}}

    # Generate report
    report = generate_procedural_report(violations, score_result, args.meeting_name)
    report_path = os.path.join(args.output_dir, "procedural_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved: {report_path}")

    # Save JSON results
    json_results = {
        "conformance_results": results,
        "compliance_score": score_result,
        "violations": violations,
        "violation_summary": summary,
    }
    json_path = os.path.join(args.output_dir, "roberts_rules_conformance.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"  JSON results saved: {json_path}")

    print(f"\n  All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
