#!/usr/bin/env python3
"""CLI runner for LLM-based structured event extraction.

Usage:
    python research/run_structured_extraction.py \
        --raw-events results/raw_events.csv \
        --agenda agenda.txt \
        --api-key sk-... \
        --output-dir results/structured
"""

import sys
import os
import argparse
import json
from unittest.mock import MagicMock

sys.modules["streamlit"] = MagicMock()

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.structured_extraction.llm_extractor import StructuredEventExtractor
from research.structured_extraction.extraction_eval import compare_extractions


def load_agenda_activities(path: str) -> list:
    """Extract activity names from agenda file."""
    import re
    activities = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+[a-z]?\.\s*", "", line)
            if cleaned and len(cleaned) > 2:
                activities.append(cleaned)
    return activities


def main():
    parser = argparse.ArgumentParser(
        description="LLM structured event extraction from meeting transcripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-events", required=True,
                        help="Path to raw_events.csv")
    parser.add_argument("--agenda", required=True,
                        help="Path to agenda text file")
    parser.add_argument("--api-key", required=True,
                        help="OpenAI API key")
    parser.add_argument("--output-dir", default="results/structured",
                        help="Directory for output files")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model name")
    parser.add_argument("--window-seconds", type=int, default=120,
                        help="Window size for extraction")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with keyword extraction")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  LLM Structured Event Extraction")
    print("=" * 60)

    # Load data
    df = pd.read_csv(args.raw_events)
    print(f"\nLoaded {len(df)} raw events from {args.raw_events}")

    agenda = load_agenda_activities(args.agenda)
    print(f"Loaded {len(agenda)} agenda activities from {args.agenda}")

    # Prepare transcript segments
    segments = []
    text_col = None
    for col in ("original_text", "details", "activity_name"):
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        print("ERROR: No text column found in raw events.")
        sys.exit(1)

    for _, row in df.iterrows():
        text = str(row.get(text_col, "")).strip()
        if text and text != "nan":
            segments.append({
                "timestamp": str(row.get("timestamp", "00:00:00")),
                "text": text,
                "activity_name": str(row.get("activity_name", "")),
            })

    print(f"  Prepared {len(segments)} transcript segments")

    # Extract
    print(f"\n--- Extracting with {args.model} ---")
    extractor = StructuredEventExtractor(api_key=args.api_key, model=args.model)
    events = extractor.extract_events(
        segments, agenda, window_seconds=args.window_seconds,
    )
    print(f"  Extracted {len(events)} structured events")

    # Save
    if events:
        events_data = [e.model_dump() if hasattr(e, "model_dump") else e for e in events]
        out_path = os.path.join(args.output_dir, "structured_events.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(events_data, f, indent=2, default=str)
        print(f"  Saved: {out_path}")

    # Compare with keyword extraction
    if args.compare:
        print("\n--- Comparison with Keyword Extraction ---")
        comparison = compare_extractions(df, events, agenda)
        comp_path = os.path.join(args.output_dir, "extraction_comparison.json")
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"  Saved: {comp_path}")

    print(f"\n  All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
