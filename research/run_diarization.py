#!/usr/bin/env python3
"""CLI runner for speaker diarization analysis.

Usage:
    python research/run_diarization.py \
        --audio-path results/meeting_audio.mp3 \
        --hf-token hf_... \
        --output-dir results/diarization \
        --merge-with results/raw_events.csv
"""

import sys
import os
import argparse
import json
from unittest.mock import MagicMock

sys.modules["streamlit"] = MagicMock()

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.diarization.diarizer import MeetingDiarizer
from research.diarization.speaker_events import (
    merge_speaker_labels,
    compute_speaker_statistics,
)
from research.diarization.turn_taking import (
    compute_turn_taking_metrics,
    detect_interruptions,
)


def main():
    parser = argparse.ArgumentParser(
        description="Speaker diarization analysis for meeting audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio-path", required=True,
                        help="Path to meeting audio file (mp3/wav)")
    parser.add_argument("--hf-token", required=True,
                        help="HuggingFace token for pyannote models")
    parser.add_argument("--output-dir", default="results/diarization",
                        help="Directory for output files")
    parser.add_argument("--whisper-model", default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="WhisperX model size")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Compute device")
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=20)
    parser.add_argument("--merge-with",
                        help="Path to existing raw_events.csv to merge speaker labels")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Speaker Diarization Analysis")
    print("=" * 60)

    # Step 1: Diarize
    print(f"\nDiarizing: {args.audio_path}")
    print(f"  Model: {args.whisper_model}, Device: {args.device}")

    diarizer = MeetingDiarizer(
        hf_token=args.hf_token,
        whisperx_model_size=args.whisper_model,
        device=args.device,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    segments = diarizer.diarize(args.audio_path)
    print(f"  Produced {len(segments)} diarized segments")

    speakers = set(s.get("speaker", "UNKNOWN") for s in segments)
    print(f"  Detected {len(speakers)} speakers: {sorted(speakers)}")

    # Save segments
    seg_df = diarizer.segments_to_dataframe(segments)
    seg_path = os.path.join(args.output_dir, "diarized_segments.csv")
    seg_df.to_csv(seg_path, index=False)
    print(f"  Saved: {seg_path}")

    # Step 2: Merge with existing events (optional)
    if args.merge_with and os.path.exists(args.merge_with):
        print(f"\n--- Merging with {args.merge_with} ---")
        existing = pd.read_csv(args.merge_with)
        merged = merge_speaker_labels(existing, segments)
        merged_path = os.path.join(args.output_dir, "speaker_events.csv")
        merged.to_csv(merged_path, index=False)
        print(f"  Merged {len(merged)} events with speaker labels")
        print(f"  Saved: {merged_path}")
        analysis_df = merged
    else:
        analysis_df = seg_df

    # Step 3: Speaker statistics
    if "speaker" in analysis_df.columns:
        print("\n--- Speaker Statistics ---")
        stats = compute_speaker_statistics(analysis_df)
        for speaker, info in stats.get("per_speaker", {}).items():
            print(f"  {speaker}: {info.get('event_count', 0)} events, "
                  f"{info.get('floor_time_seconds', 0):.0f}s floor time")

        stats_path = os.path.join(args.output_dir, "speaker_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"  Saved: {stats_path}")

        # Step 4: Turn-taking analysis
        print("\n--- Turn-Taking Metrics ---")
        metrics = compute_turn_taking_metrics(analysis_df)
        print(f"  Equity index: {metrics.get('speaker_equity_index', 'N/A'):.3f}")
        print(f"  Interruptions: {metrics.get('interruption_count', 0)}")
        print(f"  Avg turn length: {metrics.get('avg_turn_length_seconds', 0):.1f}s")
        print(f"  Dominant speaker: {metrics.get('dominant_speaker', 'N/A')}")

        interruptions = detect_interruptions(analysis_df)
        print(f"  Detected {len(interruptions)} interruptions")

        metrics_path = os.path.join(args.output_dir, "turn_taking_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Saved: {metrics_path}")

    print(f"\n  All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
