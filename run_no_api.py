#!/usr/bin/env python3
"""
LLM-free conformance pipeline for Meeting Process Twin.

Bypasses all LLM steps (no OpenAI / Ollama needed).
Uses:
  1. Regex parsing to extract agenda activities from agenda.txt
  2. PM4Py to build a sequential reference BPMN from the agenda
  3. SBERT on full transcript text to map raw events -> agenda items
  4. PM4Py token-based replay fitness + alignment diagnostics

Usage
-----
# Basic run (uses cached results/raw_events.csv):
python run_no_api.py

# Use different raw events file:
python run_no_api.py --raw-events results/raw_events.csv --agenda agenda.txt

# Sweep SBERT thresholds to find the best mapping:
python run_no_api.py --sweep

# Use a richer SBERT model (slower but more accurate):
python run_no_api.py --sbert-model all-mpnet-base-v2
"""

import sys
import os
import re
import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Mock streamlit before project imports
_mock_st = MagicMock()
_mock_st.error = lambda m: print(f"[ST ERROR] {m}", file=sys.stderr)
_mock_st.warning = lambda m: print(f"[ST WARN] {m}")
sys.modules["streamlit"] = _mock_st

import pandas as pd
import pm4py
from pm4py.objects.bpmn.obj import BPMN
from sentence_transformers import SentenceTransformer, util

from compliance_engine import ComplianceEngine


# ---------------------------------------------------------------------------
# Step 1: Parse agenda without LLM
# ---------------------------------------------------------------------------

def parse_agenda_activities(agenda_path: str) -> list:
    """
    Extract ordered activity labels from a numbered agenda text file.
    Handles both top-level items (1.) and sub-items (1a.).
    """
    with open(agenda_path, "r", encoding="utf-8") as f:
        text = f.read()

    activities = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match: "1. Title" or "1a. Title" or "10. Title" etc.
        m = re.match(r'^\d{1,2}[a-z]?\.\s+(.+)$', line)
        if m:
            label = m.group(1).strip()
            # Remove parenthetical notes for cleaner labels, but keep them for context
            activities.append(label)

    return activities


# ---------------------------------------------------------------------------
# Step 2: Build sequential BPMN from activity list (no LLM)
# ---------------------------------------------------------------------------

def build_sequential_bpmn(activities: list) -> BPMN:
    """Build a strictly sequential BPMN model: Start -> A1 -> A2 -> ... -> End."""
    bpmn_graph = BPMN()

    start_event = BPMN.StartEvent(name="Start")
    bpmn_graph.add_node(start_event)
    end_event = BPMN.EndEvent(name="End")
    bpmn_graph.add_node(end_event)

    prev = start_event
    for label in activities:
        task = BPMN.Task(name=label)
        bpmn_graph.add_node(task)
        bpmn_graph.add_flow(BPMN.SequenceFlow(prev, task))
        prev = task

    bpmn_graph.add_flow(BPMN.SequenceFlow(prev, end_event))
    return bpmn_graph


# ---------------------------------------------------------------------------
# Step 3a: Rule-based keyword mapping (imported from shared module)
# ---------------------------------------------------------------------------

from keyword_rules import KEYWORD_RULES, keyword_map, find_activity


# ---------------------------------------------------------------------------
# Step 3b: SBERT-based direct mapping using rich transcript text
# ---------------------------------------------------------------------------

def map_events_sbert(df: pd.DataFrame, activities: list, model: SentenceTransformer,
                     threshold: float = 0.40, use_keywords: bool = True) -> pd.DataFrame:
    """
    Map raw events to agenda activities.

    Strategy:
      1. Rule-based keyword matching (fast, high-confidence)
      2. SBERT cosine similarity on full transcript text (for everything else)

    Text used for SBERT: original_text > details > activity_name
    """
    if not activities:
        return df

    agenda_embeddings = model.encode(activities, convert_to_tensor=True)

    def best_text(row):
        ot = str(row.get("original_text", "")).strip()
        if ot and ot not in ("nan", ""):
            return ot[:400]
        d = str(row.get("details", "")).strip()
        if d and d not in ("nan", ""):
            return d[:200]
        return str(row.get("activity_name", "Unknown"))

    # Pre-encode all texts at once
    rows_list = [row for _, row in df.iterrows()]
    texts = [best_text(r) for r in rows_list]
    event_embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    scores_matrix = util.cos_sim(event_embeddings, agenda_embeddings)

    mapped_activities = []
    mapped_scores = []
    mapping_method = []

    for i, row in enumerate(rows_list):
        # 1. Try keyword rules first
        keyword_match = keyword_map(row, activities) if use_keywords else None
        if keyword_match:
            mapped_activities.append(keyword_match)
            mapped_scores.append(1.0)
            mapping_method.append("keyword")
            continue

        # 2. Fall back to SBERT
        scores = scores_matrix[i]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        if best_score >= threshold:
            mapped_activities.append(activities[best_idx])
            mapped_scores.append(round(best_score, 4))
            mapping_method.append("sbert")
        else:
            orig = row.get("activity_name", "Unknown")
            mapped_activities.append(f"Deviation: {orig}")
            mapped_scores.append(round(best_score, 4))
            mapping_method.append("unmatched")

    result = df.copy()
    result["mapped_activity"] = mapped_activities
    result["sbert_score"] = mapped_scores
    result["mapping_method"] = mapping_method
    return result


# ---------------------------------------------------------------------------
# Step 3c: Deduplicate trace for fitness calculation
# ---------------------------------------------------------------------------

def dedup_for_fitness(df_mapped: pd.DataFrame, activities: list) -> pd.DataFrame:
    """
    Return one row per agenda item — the FIRST occurrence in time order.
    This removes repetition noise and gives the cleanest possible conformance trace.
    Only keeps activities that appear in the agenda (skips Deviation: rows).
    """
    matched = df_mapped[~df_mapped["mapped_activity"].str.startswith("Deviation:", na=False)].copy()
    if matched.empty:
        return matched
    # Sort by timestamp BEFORE groupby to ensure first() picks chronologically earliest
    matched["__sort_ts"] = matched["timestamp"].apply(
        lambda t: sum(int(x) * m for x, m in zip(str(t).split(":"), [3600, 60, 1]))
    )
    matched = matched.sort_values("__sort_ts")
    first_occurrences = matched.groupby("mapped_activity").first().reset_index()
    first_occurrences = first_occurrences.sort_values("__sort_ts").drop(columns=["__sort_ts"])
    return first_occurrences


# ---------------------------------------------------------------------------
# PM4Py helpers
# ---------------------------------------------------------------------------

def parse_time(t_str):
    try:
        parts = str(t_str).split(":")
        if len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            total_seconds = m * 60 + s
        elif len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            total_seconds = h * 3600 + m * 60 + s
        else:
            return datetime.now()
        return datetime(2023, 1, 1) + timedelta(seconds=total_seconds)
    except Exception:
        return datetime.now()


def to_event_log(df_mapped: pd.DataFrame) -> pd.DataFrame:
    """Convert mapped events to a PM4Py-compatible event log (single case)."""
    log = df_mapped[~df_mapped["mapped_activity"].str.startswith("Deviation:", na=False)].copy()
    if log.empty:
        return None
    log["case:concept:name"] = "Meeting_1"
    log["concept:name"] = log["mapped_activity"]
    log["time:timestamp"] = log["timestamp"].apply(parse_time)
    return log


def calculate_fitness(bpmn_obj: BPMN, log_df: pd.DataFrame) -> dict:
    try:
        net, im, fm = pm4py.convert_to_petri_net(bpmn_obj)
        fitness = pm4py.fitness_token_based_replay(log_df, net, im, fm)
        alignments = pm4py.conformance_diagnostics_alignments(log_df, net, im, fm)
        return {"score": fitness["log_fitness"], "alignments": alignments}
    except Exception as e:
        print(f"  Fitness error: {e}")
        return {"score": 0.0, "alignments": []}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def analyze_mapping(df_mapped: pd.DataFrame, activities: list):
    matched = df_mapped[~df_mapped["mapped_activity"].str.startswith("Deviation:", na=False)]
    deviation = df_mapped[df_mapped["mapped_activity"].str.startswith("Deviation:", na=False)]
    match_rate = len(matched) / max(1, len(df_mapped)) * 100

    print(f"\n  Total events   : {len(df_mapped)}")
    print(f"  Matched        : {len(matched)} ({match_rate:.1f}%)")
    print(f"  Deviations     : {len(deviation)}")

    if not matched.empty:
        print(f"\n  Agenda activity coverage (top 20):")
        counts = matched["mapped_activity"].value_counts()
        for act, cnt in counts.head(20).items():
            print(f"    [{cnt:3d}x]  {act}")

        covered = set(matched["mapped_activity"].tolist())
        missing = [a for a in activities if a not in covered]
        if missing:
            print(f"\n  Agenda items with NO matched events ({len(missing)}):")
            for m in missing:
                print(f"    - {m}")

    if not deviation.empty:
        print(f"\n  Top deviation source labels:")
        dev_labels = deviation["mapped_activity"].str.replace("Deviation: ", "", regex=False)
        for lbl, cnt in dev_labels.value_counts().head(10).items():
            print(f"    [{cnt:3d}x]  {lbl}")


def analyze_alignments(alignments: list):
    deviations = []
    for align in alignments:
        for log_move, model_move in align.get("alignment", []):
            is_model_skip = model_move is None or model_move == ">>"
            if is_model_skip and log_move:
                deviations.append(log_move)

    if deviations:
        dev_counts = Counter(deviations)
        print(f"\n  Shadow/Deviation Activities ({len(dev_counts)} unique):")
        for dev, cnt in dev_counts.most_common(10):
            print(f"    [{cnt:3d}x]  {dev}")

    return deviations


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(df: pd.DataFrame, activities: list, bpmn_obj: BPMN,
                     model: SentenceTransformer, engine: ComplianceEngine,
                     thresholds: list) -> list:
    print_section("THRESHOLD SWEEP")
    print(f"  {'thresh':>6}  {'match%':>7}  {'agnd_cov%':>9}  {'fit_raw%':>8}  {'fit_dedup%':>10}  {'devs':>5}")
    results = []
    for threshold in thresholds:
        df_mapped = map_events_sbert(df, activities, model, threshold=threshold, use_keywords=True)

        matched = df_mapped[~df_mapped["mapped_activity"].str.startswith("Deviation:", na=False)]
        match_rate = len(matched) / max(1, len(df_mapped)) * 100
        covered = set(matched["mapped_activity"].tolist()) if not matched.empty else set()
        agenda_coverage = len(covered) / max(1, len(activities)) * 100

        # Raw fitness (all matched events)
        log_raw = to_event_log(df_mapped)
        fitness_raw = 0.0
        unique_devs = 0
        if log_raw is not None and not log_raw.empty:
            res = calculate_fitness(bpmn_obj, log_raw)
            fitness_raw = res.get("score", 0.0)
            for align in res.get("alignments", []):
                for log_move, model_move in align.get("alignment", []):
                    if (model_move is None or model_move == ">>") and log_move:
                        unique_devs += 1

        # Dedup fitness (first occurrence of each agenda item)
        df_dedup = dedup_for_fitness(df_mapped, activities)
        log_dedup = to_event_log(df_dedup)
        fitness_dedup = 0.0
        if log_dedup is not None and not log_dedup.empty:
            res2 = calculate_fitness(bpmn_obj, log_dedup)
            fitness_dedup = res2.get("score", 0.0)

        results.append({
            "threshold": threshold,
            "match_rate": round(match_rate, 1),
            "agenda_coverage": round(agenda_coverage, 1),
            "fitness_raw": round(fitness_raw * 100, 2),
            "fitness_dedup": round(fitness_dedup * 100, 2),
            "unique_devs": unique_devs,
            "matched_events": len(matched),
        })

        print(f"  {threshold:>6.2f}  {match_rate:>7.1f}%  {agenda_coverage:>9.1f}%  "
              f"{fitness_raw*100:>8.1f}%  {fitness_dedup*100:>10.1f}%  {unique_devs:>5}")

    # Best: maximize dedup fitness * agenda_coverage combined score
    best = max(results, key=lambda r: r["fitness_dedup"] * 0.6 + r["agenda_coverage"] * 0.4)
    print(f"\n  Best threshold by dedup-fitness: {best['threshold']} "
          f"(fit_dedup={best['fitness_dedup']}%  agenda_cov={best['agenda_coverage']}%)")
    return results, best["threshold"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # PHASE 1: Parse agenda
    print_section("PHASE 1: Agenda Parsing (No LLM)")
    activities = parse_agenda_activities(args.agenda)
    print(f"\n  Parsed {len(activities)} activities:")
    for i, a in enumerate(activities, 1):
        print(f"    {i:2d}. {a}")

    # PHASE 2: Build BPMN
    print_section("PHASE 2: Sequential BPMN Construction")
    bpmn_obj = build_sequential_bpmn(activities)
    print(f"\n  Built sequential BPMN: {len(activities)} tasks, Start -> ... -> End")

    # PHASE 3: Load raw events
    print_section("PHASE 3: Load Raw Events")
    if not os.path.isfile(args.raw_events):
        print(f"  ERROR: {args.raw_events} not found. Run test_pipeline.py first.")
        sys.exit(1)
    df = pd.read_csv(args.raw_events)
    print(f"\n  Loaded {len(df)} raw events")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Raw activity distribution (top 15):")
    for act, cnt in df["activity_name"].value_counts().head(15).items():
        print(f"    [{cnt:3d}x]  {act}")

    # PHASE 4: SBERT mapping
    print_section("PHASE 4: SBERT Model Load")
    print(f"\n  Loading: {args.sbert_model} ...")
    model = SentenceTransformer(args.sbert_model)
    print("  Model loaded.")

    engine = ComplianceEngine.__new__(ComplianceEngine)
    engine.model = model

    # Optional threshold sweep
    if args.sweep:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        sweep_results, best_threshold = sweep_thresholds(df, activities, bpmn_obj, model, engine, thresholds)
        threshold = best_threshold

        sweep_path = os.path.join(args.output_dir, "sbert_sweep.json")
        with open(sweep_path, "w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\n  Sweep saved to: {sweep_path}")
    else:
        threshold = args.threshold

    # PHASE 5: Map with chosen threshold (keyword + SBERT)
    print_section(f"PHASE 5: Keyword + SBERT Mapping (threshold={threshold})")
    df_mapped = map_events_sbert(df, activities, model, threshold=threshold, use_keywords=True)
    analyze_mapping(df_mapped, activities)

    if "mapping_method" in df_mapped.columns:
        method_counts = df_mapped["mapping_method"].value_counts()
        print(f"\n  Mapping method breakdown:")
        for method, cnt in method_counts.items():
            print(f"    {method:12s}: {cnt} events")

    mapped_path = os.path.join(args.output_dir, "no_api_mapped.csv")
    df_mapped.to_csv(mapped_path, index=False)
    print(f"\n  Saved to: {mapped_path}")

    # PHASE 6a: Raw fitness (all matched events)
    print_section("PHASE 6a: PM4Py Conformance (All Matched Events)")
    log_raw = to_event_log(df_mapped)

    if log_raw is None or log_raw.empty:
        print("\n  No matched events -- fitness cannot be calculated.")
        print("  Try lowering --threshold (e.g. --threshold 0.25)")
        return

    print(f"\n  Event log: {len(log_raw)} events in 1 trace")
    result_raw = calculate_fitness(bpmn_obj, log_raw)
    fitness_raw = result_raw.get("score", 0.0)
    alignments_raw = result_raw.get("alignments", [])
    print(f"\n  Fitness (raw, repetitions included): {fitness_raw*100:.1f}%")
    deviations_raw = analyze_alignments(alignments_raw)

    # PHASE 6b: Dedup fitness (first occurrence of each agenda item)
    print_section("PHASE 6b: PM4Py Conformance (Deduplicated Trace)")
    df_dedup = dedup_for_fitness(df_mapped, activities)
    log_dedup = to_event_log(df_dedup)

    fitness_dedup = 0.0
    deviations_dedup = []
    if log_dedup is not None and not log_dedup.empty:
        seq_preview = " -> ".join(log_dedup["concept:name"].tolist()[:8])
        print(f"\n  Deduplicated trace: {len(log_dedup)} events (first occurrence per activity)")
        print(f"  Sequence: {seq_preview} ...")
        result_dedup = calculate_fitness(bpmn_obj, log_dedup)
        fitness_dedup = result_dedup.get("score", 0.0)
        alignments_dedup = result_dedup.get("alignments", [])
        print(f"\n  *** DEDUP FITNESS: {fitness_dedup*100:.1f}% ***")
        deviations_dedup = analyze_alignments(alignments_dedup)
    else:
        print("  No deduplicated events available.")

    # Coverage report
    covered_raw = set(log_raw["concept:name"].tolist())
    print(f"\n  Agenda items covered: {len(covered_raw)}/{len(activities)}")
    for a in activities:
        status = "[OK]" if a in covered_raw else "[--]"
        print(f"    {status}  {a}")

    # Summary
    print_section("SUMMARY")
    print(f"  Raw events           : {len(df)}")
    print(f"  Mapped events        : {len(log_raw)}")
    print(f"  Deduplicated events  : {len(df_dedup) if log_dedup is not None else 0}")
    print(f"  Match rate           : {len(log_raw)/len(df)*100:.1f}%")
    print(f"  Agenda coverage      : {len(covered_raw)}/{len(activities)} ({len(covered_raw)/len(activities)*100:.1f}%)")
    print(f"  Fitness (raw)        : {fitness_raw*100:.1f}%  (all {len(log_raw)} events, repetitions counted)")
    print(f"  Fitness (dedup)      : {fitness_dedup*100:.1f}%  (first-occurrence per agenda item)")
    print(f"  Deviations (raw)     : {len(set(deviations_raw))}")
    print(f"  SBERT model          : {args.sbert_model}")
    print(f"  SBERT threshold      : {threshold}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "sbert_model": args.sbert_model,
        "sbert_threshold": threshold,
        "keyword_augmented": True,
        "raw_events": len(df),
        "mapped_events": len(log_raw),
        "deduplicated_events": len(df_dedup) if log_dedup is not None else 0,
        "match_rate": round(len(log_raw) / len(df), 4),
        "agenda_activities": len(activities),
        "agenda_coverage": len(covered_raw),
        "fitness_raw": round(fitness_raw, 4),
        "fitness_dedup": round(fitness_dedup, 4),
        "unique_deviations_raw": len(set(deviations_raw)),
    }
    summary_path = os.path.join(args.output_dir, "no_api_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-free conformance pipeline using SBERT + PM4Py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-events", default="results/raw_events.csv")
    parser.add_argument("--agenda", default="agenda.txt")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="SBERT cosine similarity threshold for mapping")
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2",
                        choices=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                        help="SBERT model to use (mpnet is more accurate but slower)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep multiple SBERT thresholds and pick the best")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
