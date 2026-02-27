#!/usr/bin/env python3
"""
Parameter sweep for Meeting Process Twin — Abstraction Layer.

Loads cached raw_events.csv (produced by test_pipeline.py or the app), then runs
the semantic abstraction + conformance pipeline across a grid of parameter combinations.
Results are saved to a CSV leaderboard so you can compare configurations easily.

Pre-requisites
--------------
1. Run test_pipeline.py once with your video to generate:
       results/raw_events.csv

2. Have your agenda ready as a text file or environment variable.

Usage
-----
python param_sweep.py \
    --api-key sk-... \
    --agenda agenda.txt \
    --raw-events results/raw_events.csv \
    --output-dir results/sweep

# Dry-run (print grid without executing):
python param_sweep.py --api-key sk-... --agenda agenda.txt \
    --raw-events results/raw_events.csv --dry-run

# Resume interrupted sweep (skips already-completed combos):
python param_sweep.py --api-key sk-... --agenda agenda.txt \
    --raw-events results/raw_events.csv --resume
"""

import sys
import os
import argparse
import csv
import json
import itertools
import time
from datetime import datetime
from unittest.mock import MagicMock

# ---- Mock streamlit before any project imports ----------------------------
_mock_st = MagicMock()
_mock_st.error = lambda m: print(f"[ST ERROR] {m}", file=sys.stderr)
_mock_st.warning = lambda m: print(f"[ST WARN] {m}")
sys.modules["streamlit"] = _mock_st

import pandas as pd
from compliance_engine import ComplianceEngine
from bpmn_gen import generate_agenda_bpmn, convert_to_event_log


# ---------------------------------------------------------------------------
# Parameter grid — edit this to add / remove combinations
# ---------------------------------------------------------------------------

PARAM_GRID = {
    # Temporal window controlling how many seconds of events are grouped per LLM call
    "window_seconds": [45, 75, 120],

    # Fraction of the window that overlaps with the next window
    "overlap_ratio": [0.3, 0.5, 0.7],

    # Minimum raw events required in a window to bother calling LLM (too few -> likely noise)
    "min_events_per_window": [3, 6, 10],

    # Minimum times a label must appear overall to be kept (denoising)
    "min_label_support": [1, 2, 3],

    # SBERT cosine similarity threshold for mapping abstracted events -> agenda items
    "sbert_threshold": [0.35, 0.45, 0.55],
}

# Fixed params (not swept)
FIXED_OPENAI_MODEL = "gpt-4o-mini"
FIXED_SHADOW_MIN_RATIO = 0.1   # single-meeting: shadow filter disabled effectively
FIXED_MAX_WINDOWS = 150
FIXED_OPENAI_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_agenda(path_or_text: str) -> str:
    if os.path.isfile(path_or_text):
        with open(path_or_text, "r", encoding="utf-8") as f:
            return f.read()
    return path_or_text


def combo_id(params: dict) -> str:
    return "__".join(f"{k}={v}" for k, v in sorted(params.items()))


def run_abstraction_and_fitness(
    df_raw: pd.DataFrame,
    activities: list,
    bpmn_obj,
    engine: ComplianceEngine,
    params: dict,
    api_key: str,
) -> dict:
    """Run one abstraction + conformance check with the given params. Returns metrics dict."""
    t0 = time.time()

    df_abstracted = engine.abstract_events_df(
        df=df_raw,
        agenda_tasks=activities,
        window_seconds=params["window_seconds"],
        overlap_ratio=params["overlap_ratio"],
        min_events_per_window=params["min_events_per_window"],
        min_label_support=params["min_label_support"],
        shadow_min_ratio=FIXED_SHADOW_MIN_RATIO,
        model="mistral",                       # Ollama fallback (usually unavailable)
        api_key=api_key,
        openai_model=FIXED_OPENAI_MODEL,
        openai_timeout=FIXED_OPENAI_TIMEOUT,
        max_windows_per_run=FIXED_MAX_WINDOWS,
        cache={},
        debug_callback=None,
    )

    abstraction_time = time.time() - t0

    if df_abstracted.empty:
        return {
            "abstracted_events": 0,
            "mapped_events": 0,
            "fitness_score": 0.0,
            "unique_deviations": 0,
            "shadow_events": 0,
            "formal_events": 0,
            "noise_ratio": 1.0,
            "abstraction_time_s": round(abstraction_time, 1),
            "error": "all_noise",
        }

    # Mapping
    df_mapped = engine.map_events_to_agenda(
        df_abstracted, activities, threshold=params["sbert_threshold"]
    )

    # Count formal vs shadow
    formal_mask = ~df_abstracted["activity_name"].str.startswith("Shadow:", na=False)
    shadow_count = int((~formal_mask).sum())
    formal_count = int(formal_mask.sum())

    # Fitness
    log_data = convert_to_event_log(df_mapped)
    fitness = 0.0
    unique_deviations = 0
    if log_data is not None and bpmn_obj is not None:
        try:
            result = engine.calculate_fitness(bpmn_obj, log_data)
            fitness = result.get("score", 0.0)
            alignments = result.get("alignments", [])
            devs = set()
            for align in alignments:
                for log_move, model_move in align.get("alignment", []):
                    if (model_move is None or model_move == ">>") and log_move:
                        devs.add(log_move)
            unique_deviations = len(devs)
        except Exception as e:
            print(f"    Fitness error: {e}")

    noise_ratio = 1.0 - (len(df_abstracted) / max(1, len(df_raw)))

    return {
        "abstracted_events": len(df_abstracted),
        "mapped_events": len(df_mapped),
        "fitness_score": round(fitness, 4),
        "unique_deviations": unique_deviations,
        "shadow_events": shadow_count,
        "formal_events": formal_count,
        "noise_ratio": round(noise_ratio, 3),
        "abstraction_time_s": round(abstraction_time, 1),
        "error": "",
    }


# ---------------------------------------------------------------------------
# Main sweep logic
# ---------------------------------------------------------------------------

def run_sweep(args):
    os.makedirs(args.output_dir, exist_ok=True)
    results_csv = os.path.join(args.output_dir, "sweep_results.csv")

    # Load raw events
    if not os.path.isfile(args.raw_events):
        print(f"ERROR: raw_events not found: {args.raw_events}", file=sys.stderr)
        sys.exit(1)
    df_raw = pd.read_csv(args.raw_events)
    print(f"Loaded {len(df_raw)} raw events from: {args.raw_events}")

    # Load agenda + build BPMN (once)
    agenda_text = load_agenda(args.agenda)
    print("Generating agenda BPMN (once)...")
    bpmn_viz, activities, bpmn_obj = generate_agenda_bpmn(agenda_text, args.api_key)
    print(f"Agenda activities ({len(activities)}): {activities}")

    engine = ComplianceEngine()

    # Build full grid
    keys = list(PARAM_GRID.keys())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*[PARAM_GRID[k] for k in keys])]
    total = len(all_combos)
    print(f"\nParameter grid: {total} combinations to test\n")

    if args.dry_run:
        for i, combo in enumerate(all_combos, 1):
            print(f"  [{i:3d}] {combo}")
        return

    # Load already-completed combos (for --resume)
    completed_ids = set()
    if args.resume and os.path.isfile(results_csv):
        with open(results_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_ids.add(row.get("combo_id", ""))
        print(f"Resuming: {len(completed_ids)} combos already done, skipping them.\n")

    # Open CSV for appending
    fieldnames = (
        ["combo_id", "run_timestamp"]
        + keys
        + ["abstracted_events", "mapped_events", "fitness_score",
           "unique_deviations", "shadow_events", "formal_events",
           "noise_ratio", "abstraction_time_s", "error"]
    )

    csv_exists = os.path.isfile(results_csv) and args.resume
    csv_file = open(results_csv, "a" if args.resume else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()

    try:
        for i, combo in enumerate(all_combos, 1):
            cid = combo_id(combo)
            if cid in completed_ids:
                print(f"[{i:3d}/{total}] SKIP (already done): {combo}")
                continue

            print(f"[{i:3d}/{total}] Testing: {combo}")
            try:
                metrics = run_abstraction_and_fitness(
                    df_raw=df_raw,
                    activities=activities,
                    bpmn_obj=bpmn_obj,
                    engine=engine,
                    params=combo,
                    api_key=args.api_key,
                )
            except Exception as e:
                metrics = {k: "" for k in fieldnames}
                metrics["error"] = str(e)[:120]
                print(f"    ERROR: {e}")

            row = {
                "combo_id": cid,
                "run_timestamp": datetime.now().isoformat(timespec="seconds"),
                **combo,
                **metrics,
            }
            writer.writerow(row)
            csv_file.flush()

            print(
                f"    fitness={metrics.get('fitness_score', '?'):.4f}  "
                f"abstracted={metrics.get('abstracted_events', '?')}  "
                f"shadow={metrics.get('shadow_events', '?')}  "
                f"noise_ratio={metrics.get('noise_ratio', '?')}  "
                f"deviations={metrics.get('unique_deviations', '?')}  "
                f"time={metrics.get('abstraction_time_s', '?')}s"
            )

    finally:
        csv_file.close()

    print(f"\nSweep complete. Results saved to: {results_csv}")

    # Print leaderboard
    print_leaderboard(results_csv)


def print_leaderboard(csv_path: str, top_n: int = 10):
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Score = fitness (primary), then lower deviations (secondary), then more abstracted events
    df["fitness_score"] = pd.to_numeric(df["fitness_score"], errors="coerce").fillna(0)
    df["unique_deviations"] = pd.to_numeric(df["unique_deviations"], errors="coerce").fillna(999)
    df["abstracted_events"] = pd.to_numeric(df["abstracted_events"], errors="coerce").fillna(0)

    df_sorted = df.sort_values(
        by=["fitness_score", "unique_deviations", "abstracted_events"],
        ascending=[False, True, False],
    )

    print(f"\n{'='*70}")
    print(f"  TOP {top_n} CONFIGURATIONS (by fitness score)")
    print(f"{'='*70}")
    param_cols = list(PARAM_GRID.keys())
    for rank, (_, row) in enumerate(df_sorted.head(top_n).iterrows(), 1):
        params_str = "  ".join(f"{k}={row[k]}" for k in param_cols)
        print(
            f"  #{rank:2d}  fitness={row['fitness_score']:.4f}  devs={int(row['unique_deviations'])}  "
            f"abstracted={int(row['abstracted_events'])}  shadow={int(row.get('shadow_events', 0))}\n"
            f"       {params_str}"
        )
    print()

    # Save top-10 to separate JSON
    top_path = os.path.join(os.path.dirname(csv_path), "top_configs.json")
    top_records = df_sorted.head(top_n).to_dict(orient="records")
    with open(top_path, "w", encoding="utf-8") as f:
        json.dump(top_records, f, indent=2, default=str)
    print(f"  Top configs saved to: {top_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for Meeting Process Twin abstraction layer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                        help="OpenAI API key")
    parser.add_argument("--agenda", required=True,
                        help="Path to agenda text file OR raw agenda text")
    parser.add_argument("--raw-events", default="results/raw_events.csv",
                        help="Path to cached raw_events.csv from test_pipeline.py")
    parser.add_argument("--output-dir", default="results/sweep",
                        help="Directory to write sweep_results.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the parameter grid without running anything")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed combinations (append to existing CSV)")
    parser.add_argument("--leaderboard-only", action="store_true",
                        help="Just print leaderboard from existing sweep_results.csv")
    args = parser.parse_args()

    if args.leaderboard_only:
        results_csv = os.path.join(args.output_dir, "sweep_results.csv")
        if not os.path.isfile(results_csv):
            print(f"No results file found: {results_csv}", file=sys.stderr)
            sys.exit(1)
        print_leaderboard(results_csv)
        return

    run_sweep(args)


if __name__ == "__main__":
    main()
