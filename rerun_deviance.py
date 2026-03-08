"""Re-run deviance classification on all meetings using the fixed key mapping.

Reads each meeting's mapped_events.csv, runs the DevianceClassifier (rules-only,
no API key needed), and patches the 'deviance' section of conformance.json in-place.
"""

import json
import os
import sys

import pandas as pd

from research.deviance.deviance_classifier import DevianceClassifier


MEETINGS_DIR = os.path.join(os.path.dirname(__file__), "meetings")


def rerun_meeting(meeting_dir: str) -> dict | None:
    """Re-run deviance classification for a single meeting."""
    conf_path = os.path.join(meeting_dir, "conformance.json")
    mapped_path = os.path.join(meeting_dir, "variant_llm", "mapped_events.csv")

    if not os.path.exists(conf_path) or not os.path.exists(mapped_path):
        return None

    # Load mapped events
    df = pd.read_csv(mapped_path)

    # Load existing declare violations from conformance.json
    with open(conf_path, "r") as f:
        conf = json.load(f)

    declare_violations = conf.get("declare", {}).get("violations", [])

    # Run deviance classifier (rules-only, no API key)
    classifier = DevianceClassifier(api_key=None)
    classified = classifier.classify(df, declare_violations)
    summary = classifier.generate_deviance_summary(classified)

    cat_dist = summary.get("category_distribution", {})
    new_deviance = {
        "benign": cat_dist.get("benign", 0),
        "violation": cat_dist.get("violation", 0),
        "efficiency": cat_dist.get("efficient", 0),
        "innovation": cat_dist.get("innovation", 0),
        "disruption": cat_dist.get("disruption", 0),
        "unknown": cat_dist.get("unknown", 0),
        "severity_pct": round(summary.get("severity_score", 0.0), 2),
    }

    # Patch conformance.json
    old_deviance = conf.get("deviance", {})
    conf["deviance"] = new_deviance

    with open(conf_path, "w") as f:
        json.dump(conf, f, indent=2)

    return {
        "meeting": os.path.basename(meeting_dir),
        "old": old_deviance,
        "new": new_deviance,
    }


def main():
    if not os.path.isdir(MEETINGS_DIR):
        print(f"ERROR: meetings directory not found: {MEETINGS_DIR}")
        sys.exit(1)

    meeting_dirs = sorted([
        os.path.join(MEETINGS_DIR, d)
        for d in os.listdir(MEETINGS_DIR)
        if os.path.isdir(os.path.join(MEETINGS_DIR, d))
    ])

    print(f"Found {len(meeting_dirs)} meeting directories.\n")

    results = []
    changed = 0

    for mdir in meeting_dirs:
        name = os.path.basename(mdir)
        result = rerun_meeting(mdir)
        if result is None:
            print(f"  SKIP  {name} (missing files)")
            continue

        old = result["old"]
        new = result["new"]
        diff = old != new
        if diff:
            changed += 1

        marker = "CHANGED" if diff else "same"
        total = sum(v for k, v in new.items() if k != "severity_pct")
        print(f"  {marker:7s}  {name:40s}  "
              f"ben={new['benign']:3d}  viol={new['violation']:3d}  "
              f"eff={new['efficiency']:3d}  innov={new['innovation']:3d}  "
              f"disr={new['disruption']:3d}  unk={new['unknown']:3d}  "
              f"total={total}")
        results.append(result)

    print(f"\n{'='*80}")
    print(f"Processed: {len(results)} meetings  |  Changed: {changed}")

    # Aggregate stats
    totals = {"benign": 0, "violation": 0, "efficiency": 0,
              "innovation": 0, "disruption": 0, "unknown": 0}
    for r in results:
        for k in totals:
            totals[k] += r["new"][k]

    grand_total = sum(totals.values())
    print(f"\nAggregate deviance distribution across {len(results)} meetings:")
    for k, v in totals.items():
        pct = v / grand_total * 100 if grand_total > 0 else 0
        print(f"  {k:12s}: {v:5d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':12s}: {grand_total:5d}")


if __name__ == "__main__":
    main()
