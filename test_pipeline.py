#!/usr/bin/env python3
"""
Headless pipeline test for Meeting Process Twin.

Runs the full pipeline without Streamlit so you can test from the command line,
cache expensive Whisper results, and iterate quickly on abstraction parameters.

Usage examples
--------------
# First run: transcribe + save raw events
python test_pipeline.py --api-key sk-... --video council_012026_2022605.mp4 \
    --agenda agenda.txt --output-dir ./results

# Fast re-run using cached transcription (skip Whisper + MediaPipe):
python test_pipeline.py --api-key sk-... --video council_012026_2022605.mp4 \
    --agenda agenda.txt --output-dir ./results \
    --skip-video-processing \
    --window-seconds 90 --overlap-ratio 0.4 --min-events 8 --min-label-support 2

# Using OpenAI for abstraction (no local Ollama):
python test_pipeline.py --api-key sk-... --video council_012026_2022605.mp4 \
    --agenda agenda.txt --output-dir ./results --skip-video-processing \
    --openai-abstraction-model gpt-4o-mini
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock Streamlit BEFORE importing any project module that touches it
# ---------------------------------------------------------------------------
_mock_progress = MagicMock()
_mock_progress.progress = MagicMock()
_mock_progress.empty = MagicMock()

_mock_st = MagicMock()
_mock_st.progress = MagicMock(return_value=_mock_progress)
_mock_st.error = lambda msg: print(f"[ST ERROR] {msg}", file=sys.stderr)
_mock_st.warning = lambda msg: print(f"[ST WARNING] {msg}")

sys.modules["streamlit"] = _mock_st

# Now safe to import project modules
import pandas as pd
from video_processor import VideoProcessor
from compliance_engine import ComplianceEngine
from bpmn_gen import generate_agenda_bpmn, convert_to_event_log, generate_discovered_bpmn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_agenda(path_or_text: str) -> str:
    if os.path.isfile(path_or_text):
        with open(path_or_text, "r", encoding="utf-8") as f:
            return f.read()
    return path_or_text


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_metrics(label: str, df: pd.DataFrame):
    if df is None or df.empty:
        print(f"  {label}: <empty>")
        return
    print(f"  {label}: {len(df)} rows")
    if "activity_name" in df.columns:
        counts = df["activity_name"].value_counts()
        for act, cnt in counts.head(15).items():
            print(f"    [{cnt:3d}x]  {act}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    os.makedirs(args.output_dir, exist_ok=True)
    raw_events_cache = os.path.join(args.output_dir, "raw_events.csv")

    # ---- PHASE 1: Video Processing ----------------------------------------
    print_section("PHASE 1: Video Processing")

    if args.skip_video_processing and os.path.exists(raw_events_cache):
        print(f"  Loading cached raw events from: {raw_events_cache}")
        df_raw = pd.read_csv(raw_events_cache)
        print(f"  Loaded {len(df_raw)} events.")
    else:
        if not args.api_key:
            print("ERROR: --api-key is required for video processing.", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(args.video):
            print(f"ERROR: Video not found: {args.video}", file=sys.stderr)
            sys.exit(1)

        print(f"  Processing video: {args.video}")
        if getattr(args, 'local_whisper', False):
            print(f"  Using local Whisper model: {args.local_whisper_model}")
        t0 = time.time()
        processor = VideoProcessor(args.api_key, debug=args.debug)
        df_raw = processor.process_video(
            args.video,
            use_local_whisper=getattr(args, 'local_whisper', False),
            local_whisper_model=getattr(args, 'local_whisper_model', 'base'),
        )
        elapsed = time.time() - t0

        # SAVE FIRST — before any print that could fail on unicode
        df_raw.to_csv(raw_events_cache, index=False)

        print(f"  Done in {elapsed:.1f}s -> {len(df_raw)} events extracted.")
        print(f"  Saved to: {raw_events_cache}")

    print_metrics("Raw events", df_raw)

    # ---- PHASE 2: Agenda BPMN ---------------------------------------------
    print_section("PHASE 2: Agenda BPMN Generation")

    agenda_text = load_agenda(args.agenda)
    if not agenda_text.strip():
        print("ERROR: Agenda is empty.", file=sys.stderr)
        sys.exit(1)

    if not args.api_key:
        print("ERROR: --api-key required for BPMN generation.", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    bpmn_viz, activities, bpmn_obj = generate_agenda_bpmn(agenda_text, args.api_key)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s -> {len(activities)} agenda activities:")
    for a in activities:
        print(f"    - {a}")

    # ---- PHASE 3: Semantic Abstraction ------------------------------------
    print_section("PHASE 3: Semantic Abstraction")

    engine = ComplianceEngine()

    t0 = time.time()
    df_abstracted = engine.abstract_events_df(
        df=df_raw,
        agenda_tasks=activities,
        window_seconds=args.window_seconds,
        overlap_ratio=args.overlap_ratio,
        min_events_per_window=args.min_events,
        min_label_support=args.min_label_support,
        shadow_min_ratio=args.shadow_min_ratio,
        model=args.ollama_model,
        api_key=args.api_key,
        openai_model=args.openai_abstraction_model,
        openai_timeout=args.openai_timeout,
        max_windows_per_run=args.max_windows,
        cache={},
        debug_callback=print if args.debug else None,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s -> {len(df_abstracted)} abstracted events.")
    print_metrics("Abstracted events", df_abstracted)

    abstracted_cache = os.path.join(args.output_dir, "abstracted_events.csv")
    if not df_abstracted.empty:
        df_abstracted.to_csv(abstracted_cache, index=False)
        print(f"  Saved to: {abstracted_cache}")

    # ---- PHASE 4: Event Mapping -------------------------------------------
    print_section("PHASE 4: Agenda Mapping (SBERT)")

    if df_abstracted.empty or not activities or bpmn_obj is None:
        print("  Skipping mapping: no abstracted events or BPMN model.")
        return

    t0 = time.time()
    df_mapped = engine.map_events_to_agenda(df_abstracted, activities, threshold=args.sbert_threshold)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s -> {len(df_mapped)} mapped events.")
    _pm = df_mapped.copy()
    if "mapped_activity" in _pm.columns:
        _pm["activity_name"] = _pm["mapped_activity"]
    print_metrics("Mapped activities", _pm)

    mapped_cache = os.path.join(args.output_dir, "mapped_events.csv")
    df_mapped.to_csv(mapped_cache, index=False)
    print(f"  Saved to: {mapped_cache}")

    # ---- PHASE 5: Fitness Score -------------------------------------------
    print_section("PHASE 5: Conformance Checking")

    log_data = convert_to_event_log(df_mapped)
    if log_data is None:
        print("  Could not build event log for fitness calculation.")
        return

    t0 = time.time()
    result = engine.calculate_fitness(bpmn_obj, log_data)
    elapsed = time.time() - t0

    fitness = result.get("score", 0.0)
    alignments = result.get("alignments", [])
    print(f"  Done in {elapsed:.1f}s")
    print(f"\n  *** FITNESS SCORE: {fitness*100:.1f}% ***")
    print(f"  Alignments: {len(alignments)} trace(s)")

    # Deviation summary
    deviations = []
    for align in alignments:
        for log_move, model_move in align.get("alignment", []):
            is_model_skip = model_move is None or model_move == ">>"
            if is_model_skip and log_move:
                deviations.append(log_move)

    if deviations:
        from collections import Counter
        dev_counts = Counter(deviations)
        print(f"\n  Shadow Activities / Deviations detected ({len(dev_counts)} unique):")
        for dev, cnt in dev_counts.most_common(10):
            print(f"    [{cnt:3d}x]  {dev}")

    # ---- PHASE 5b: Dedup Fitness (first-occurrence per agenda item) ---------
    print_section("PHASE 5b: Conformance Checking (Deduplicated Trace)")
    try:
        from datetime import timedelta
        _matched = df_mapped[~df_mapped.get("mapped_activity", df_mapped.get("activity_name", pd.Series())).str.startswith("Deviation:", na=False)].copy()
        _act_col = "mapped_activity" if "mapped_activity" in _matched.columns else "activity_name"
        if not _matched.empty:
            # Sort by timestamp BEFORE groupby to ensure first() picks chronologically earliest
            _matched["__ts"] = _matched["timestamp"].apply(
                lambda t: sum(int(x)*m for x, m in zip(str(t).split(":"), [3600,60,1])))
            _matched = _matched.sort_values("__ts")
            _first = _matched.groupby(_act_col).first().reset_index()
            _first = _first.sort_values("__ts").drop(columns=["__ts"])
            _first["activity_name"] = _first[_act_col]
            _log_dedup = convert_to_event_log(_first)
            if _log_dedup is not None and not _log_dedup.empty:
                _seq = " -> ".join(_log_dedup["concept:name"].tolist()[:8])
                print(f"  Dedup trace: {len(_log_dedup)} events")
                print(f"  Sequence: {_seq} ...")
                _r2 = engine.calculate_fitness(bpmn_obj, _log_dedup)
                _fd = _r2.get("score", 0.0)
                print(f"\n  *** DEDUP FITNESS: {_fd*100:.1f}% ***")
                _cov = set(_log_dedup["concept:name"].tolist())
                print(f"  Agenda coverage: {len(_cov)}/{len(activities)}")
                for _a in activities:
                    print(f"    {'[OK]' if _a in _cov else '[--]'}  {_a}")
    except Exception as e:
        print(f"  Dedup fitness error: {e}")

    # ---- PHASE 6: Process Discovery ----------------------------------------
    print_section("PHASE 6: Process Discovery (Inductive Miner)")

    graph, evidence_map = generate_discovered_bpmn(log_data)
    if graph:
        print(f"  BPMN discovered -> {len(evidence_map)} activities in evidence map.")
        for act, ev in list(evidence_map.items())[:5]:
            print(f"    {act}: {ev[:80]}")
    else:
        print("  Not enough data to discover BPMN structure.")

    # ---- Summary ----------------------------------------------------------
    print_section("SUMMARY")
    print(f"  Raw events:        {len(df_raw)}")
    print(f"  Abstracted events: {len(df_abstracted)}")
    print(f"  Mapped events:     {len(df_mapped)}")
    print(f"  Fitness score:     {fitness*100:.1f}%")
    print(f"  Deviations:        {len(set(deviations))}")
    print(f"\n  Parameters used:")
    print(f"    window_seconds      = {args.window_seconds}")
    print(f"    overlap_ratio       = {args.overlap_ratio}")
    print(f"    min_events          = {args.min_events}")
    print(f"    min_label_support   = {args.min_label_support}")
    print(f"    shadow_min_ratio    = {args.shadow_min_ratio}")
    print(f"    sbert_threshold     = {args.sbert_threshold}")
    print(f"    openai_model        = {args.openai_abstraction_model}")

    # Save summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "window_seconds": args.window_seconds,
            "overlap_ratio": args.overlap_ratio,
            "min_events": args.min_events,
            "min_label_support": args.min_label_support,
            "shadow_min_ratio": args.shadow_min_ratio,
            "sbert_threshold": args.sbert_threshold,
            "openai_abstraction_model": args.openai_abstraction_model,
        },
        "metrics": {
            "raw_events": len(df_raw),
            "abstracted_events": len(df_abstracted),
            "mapped_events": len(df_mapped),
            "fitness_score": round(fitness, 4),
            "unique_deviations": len(set(deviations)),
            "agenda_activities": len(activities),
        },
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Headless Meeting Process Twin pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    parser.add_argument("--video", required=True, help="Path to the meeting video (.mp4)")
    parser.add_argument("--agenda", required=True, help="Path to agenda text file OR raw agenda text")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key")

    # Output
    parser.add_argument("--output-dir", default="./results", help="Directory to save results")

    # Processing control
    parser.add_argument("--skip-video-processing", action="store_true",
                        help="Load cached raw_events.csv instead of re-running Whisper+MediaPipe")
    parser.add_argument("--local-whisper", action="store_true",
                        help="Use local openai-whisper model instead of API (no API cost)")
    parser.add_argument("--local-whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Local Whisper model size (larger=more accurate but slower)")
    parser.add_argument("--debug", action="store_true", help="Verbose debug output")

    # Abstraction parameters
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--min-label-support", type=int, default=2)
    parser.add_argument("--shadow-min-ratio", type=float, default=0.15)
    parser.add_argument("--max-windows", type=int, default=150)

    # Model selection
    parser.add_argument("--ollama-model", default="mistral")
    parser.add_argument("--openai-abstraction-model", default="gpt-4o-mini")
    parser.add_argument("--openai-timeout", type=int, default=60)

    # Mapping parameter
    parser.add_argument("--sbert-threshold", type=float, default=0.45,
                        help="SBERT cosine similarity threshold for agenda mapping")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
