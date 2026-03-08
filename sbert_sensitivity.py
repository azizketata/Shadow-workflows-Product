"""SBERT threshold sensitivity analysis across all 54 meetings.

For each meeting, loads the abstracted events (LLM output) and agenda,
then re-runs SBERT mapping at multiple thresholds to measure how
threshold choice affects fitness, shadow %, and agenda coverage.

Output: thesis/figures/sbert_sensitivity.png + JSON results.
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Local imports
from pm4py.objects.bpmn.obj import BPMN
from bpmn_gen import convert_to_event_log
from compliance_engine import ComplianceEngine
from pipeline.time_utils import ts_to_seconds
from keyword_rules import keyword_map

MEETINGS_DIR = os.path.join(os.path.dirname(__file__), "meetings")
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def _build_sequential_bpmn(activities: list[str]) -> BPMN:
    """Build a simple sequential BPMN model from an activity list (no API needed)."""
    bpmn_graph = BPMN()
    start = BPMN.StartEvent(name="Start")
    end = BPMN.EndEvent(name="End")
    bpmn_graph.add_node(start)
    bpmn_graph.add_node(end)

    prev = start
    for act in activities:
        task = BPMN.Task(name=act)
        bpmn_graph.add_node(task)
        bpmn_graph.add_flow(BPMN.SequenceFlow(prev, task))
        prev = task
    bpmn_graph.add_flow(BPMN.SequenceFlow(prev, end))
    return bpmn_graph


def load_meeting_data(meeting_dir: str) -> dict | None:
    """Load abstracted events and agenda for a meeting."""
    abstracted_path = os.path.join(meeting_dir, "variant_llm", "abstracted_events.csv")
    agenda_path = os.path.join(meeting_dir, "agenda.txt")
    conf_path = os.path.join(meeting_dir, "conformance.json")

    if not all(os.path.exists(p) for p in [abstracted_path, agenda_path, conf_path]):
        return None

    df = pd.read_csv(abstracted_path)
    if df.empty:
        return None

    with open(agenda_path, "r", encoding="utf-8") as f:
        activities = [line.strip() for line in f if line.strip()]

    with open(conf_path, "r") as f:
        conf = json.load(f)

    return {
        "df": df,
        "activities": activities,
        "meeting_id": conf.get("meeting_id", os.path.basename(meeting_dir)),
        "city": conf.get("city", "Unknown"),
    }


def map_at_threshold(
    df: pd.DataFrame,
    activities: list[str],
    sbert_model: SentenceTransformer,
    agenda_embeddings,
    threshold: float,
) -> pd.DataFrame:
    """Re-run SBERT mapping at a specific threshold (fast — reuses precomputed embeddings)."""
    mapped_df = df.copy()
    target_col = "activity_name" if "activity_name" in mapped_df.columns else "concept:name"
    unique_labels = mapped_df[target_col].unique().tolist()

    # Pass 1: Keyword rules
    keyword_matches = {}
    for label in unique_labels:
        sample_rows = mapped_df[mapped_df[target_col] == label]
        if not sample_rows.empty:
            km = keyword_map(sample_rows.iloc[0].to_dict(), activities)
            if km:
                keyword_matches[label] = km

    # Pass 2: SBERT for remaining
    sbert_labels = [l for l in unique_labels if l not in keyword_matches]
    label_map = dict(keyword_matches)

    if sbert_labels:
        enriched_texts = []
        for label in sbert_labels:
            sample_rows = mapped_df[mapped_df[target_col] == label]
            if not sample_rows.empty and "original_text" in mapped_df.columns:
                orig = str(sample_rows.iloc[0].get("original_text", "")).strip()
                if orig and orig != "nan":
                    enriched_texts.append(f"{label}: {orig[:100]}")
                else:
                    details = str(sample_rows.iloc[0].get("details", "")).strip()
                    if details and details != "nan":
                        enriched_texts.append(f"{label}: {details[:100]}")
                    else:
                        enriched_texts.append(label)
            else:
                enriched_texts.append(label)

        label_embeddings = sbert_model.encode(enriched_texts, convert_to_tensor=True)
        scores_matrix = util.cos_sim(label_embeddings, agenda_embeddings)

        for i, label in enumerate(sbert_labels):
            scores = scores_matrix[i]
            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])
            if best_score >= threshold:
                label_map[label] = activities[best_idx]
            else:
                label_map[label] = f"Deviation: {label}"

    mapped_df["mapped_activity"] = mapped_df[target_col].map(label_map)
    mapped_df["concept:name"] = mapped_df["mapped_activity"]
    return mapped_df


def compute_metrics(df_mapped: pd.DataFrame, activities: list[str], bpmn_obj, engine) -> dict:
    """Compute fitness, shadow %, and coverage from mapped events."""
    act_col = "mapped_activity"
    shadow_mask = df_mapped[act_col].str.startswith("Deviation:", na=False)
    formal = df_mapped[~shadow_mask]
    shadow_count = int(shadow_mask.sum())
    formal_count = int((~shadow_mask).sum())
    total = formal_count + shadow_count

    # Match rate
    match_rate = formal_count / total if total > 0 else 0.0

    # Agenda coverage
    covered = set(formal[act_col].unique()) if not formal.empty else set()
    coverage_pct = len(covered) / len(activities) if activities else 0.0

    # Shadow %
    shadow_pct = shadow_count / total if total > 0 else 0.0

    # Dedup fitness (first occurrence per agenda item)
    dedup_fitness = 0.0
    if not formal.empty and bpmn_obj is not None:
        try:
            formal_copy = formal.copy()
            formal_copy["__ts"] = formal_copy["timestamp"].apply(ts_to_seconds)
            formal_copy = formal_copy.sort_values("__ts")
            first = formal_copy.groupby(act_col).first().reset_index()
            first = first.sort_values("__ts").drop(columns=["__ts"])
            first["activity_name"] = first[act_col]

            log_dedup = convert_to_event_log(first)
            if log_dedup is not None and not log_dedup.empty:
                fit_val = engine.calculate_fitness(bpmn_obj, log_dedup)
                if fit_val is not None:
                    dedup_fitness = fit_val.get("score", 0.0)
        except Exception as e:
            pass  # Keep 0.0

    return {
        "match_rate": round(match_rate, 4),
        "shadow_pct": round(shadow_pct, 4),
        "coverage_pct": round(coverage_pct, 4),
        "coverage_count": len(covered),
        "dedup_fitness": round(dedup_fitness, 4),
        "formal_count": formal_count,
        "shadow_count": shadow_count,
    }


def main():
    t_start = time.time()

    # Load SBERT model once
    sbert_model_name = "all-MiniLM-L6-v2"
    print(f"Loading SBERT model: {sbert_model_name}")
    sbert_model = SentenceTransformer(sbert_model_name)

    # Initialize compliance engine (for fitness computation)
    engine = ComplianceEngine.__new__(ComplianceEngine)
    engine.model = sbert_model

    # Find all meetings
    meeting_dirs = sorted([
        os.path.join(MEETINGS_DIR, d)
        for d in os.listdir(MEETINGS_DIR)
        if os.path.isdir(os.path.join(MEETINGS_DIR, d)) and d != "_aggregated"
    ])

    print(f"Found {len(meeting_dirs)} meeting directories.")
    print(f"Thresholds: {THRESHOLDS}\n")

    # Results: threshold -> list of per-meeting metrics
    all_results = {t: [] for t in THRESHOLDS}
    skipped = 0

    for i, mdir in enumerate(meeting_dirs):
        name = os.path.basename(mdir)
        data = load_meeting_data(mdir)
        if data is None:
            skipped += 1
            continue

        df = data["df"]
        activities = data["activities"]

        # Build simple sequential BPMN from activity list (no API needed)
        try:
            bpmn_obj = _build_sequential_bpmn(activities)
        except Exception:
            bpmn_obj = None

        # Precompute agenda embeddings once per meeting
        agenda_embeddings = sbert_model.encode(activities, convert_to_tensor=True)

        print(f"  [{i+1:2d}/{len(meeting_dirs)}] {name:40s} ({len(df)} events, {len(activities)} agenda items)", end="")

        for threshold in THRESHOLDS:
            df_mapped = map_at_threshold(df, activities, sbert_model, agenda_embeddings, threshold)
            metrics = compute_metrics(df_mapped, activities, bpmn_obj, engine)
            metrics["meeting_id"] = data["meeting_id"]
            metrics["city"] = data["city"]
            metrics["threshold"] = threshold
            all_results[threshold].append(metrics)

        print("  done")

    elapsed = time.time() - t_start
    print(f"\nProcessed {len(meeting_dirs) - skipped} meetings in {elapsed:.0f}s (skipped {skipped})")

    # Aggregate results
    summary = []
    for t in THRESHOLDS:
        if not all_results[t]:
            continue
        fitness_vals = [m["dedup_fitness"] for m in all_results[t]]
        shadow_vals = [m["shadow_pct"] for m in all_results[t]]
        coverage_vals = [m["coverage_pct"] for m in all_results[t]]
        match_vals = [m["match_rate"] for m in all_results[t]]

        row = {
            "threshold": t,
            "mean_fitness": round(np.mean(fitness_vals), 4),
            "median_fitness": round(np.median(fitness_vals), 4),
            "std_fitness": round(np.std(fitness_vals), 4),
            "mean_shadow_pct": round(np.mean(shadow_vals), 4),
            "median_shadow_pct": round(np.median(shadow_vals), 4),
            "mean_coverage_pct": round(np.mean(coverage_vals), 4),
            "median_coverage_pct": round(np.median(coverage_vals), 4),
            "mean_match_rate": round(np.mean(match_vals), 4),
        }
        summary.append(row)

    # Print summary table
    print(f"\n{'Threshold':>9}  {'Fitness':>8}  {'Shadow%':>8}  {'Coverage%':>10}  {'MatchRate':>10}")
    print("-" * 55)
    for r in summary:
        print(f"  {r['threshold']:.2f}     {r['mean_fitness']:.3f}     {r['mean_shadow_pct']*100:.1f}%     "
              f"{r['mean_coverage_pct']*100:.1f}%       {r['mean_match_rate']*100:.1f}%")

    # Save JSON results
    output_dir = os.path.join(os.path.dirname(__file__), "thesis", "figures")
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "sbert_sensitivity.json")
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "per_meeting": all_results}, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Generate figure
    generate_figure(summary, all_results, output_dir)


def generate_figure(summary: list[dict], all_results: dict, output_dir: str):
    """Generate the sensitivity analysis figure."""
    thresholds = [r["threshold"] for r in summary]
    mean_fitness = [r["mean_fitness"] for r in summary]
    mean_shadow = [r["mean_shadow_pct"] * 100 for r in summary]
    mean_coverage = [r["mean_coverage_pct"] * 100 for r in summary]
    mean_match = [r["mean_match_rate"] * 100 for r in summary]

    # Also compute per-meeting spreads for confidence bands
    fitness_all = [[m["dedup_fitness"] for m in all_results[t]] for t in thresholds]
    shadow_all = [[m["shadow_pct"] * 100 for m in all_results[t]] for t in thresholds]
    coverage_all = [[m["coverage_pct"] * 100 for m in all_results[t]] for t in thresholds]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    colors = {"fitness": "#1976D2", "shadow": "#E53935", "coverage": "#43A047"}

    # Panel 1: Dedup Fitness
    ax = axes[0]
    q25 = [np.percentile(f, 25) for f in fitness_all]
    q75 = [np.percentile(f, 75) for f in fitness_all]
    ax.fill_between(thresholds, q25, q75, alpha=0.2, color=colors["fitness"])
    ax.plot(thresholds, mean_fitness, "o-", color=colors["fitness"], linewidth=2, markersize=5)
    ax.set_xlabel("SBERT Cosine Similarity Threshold", fontsize=10)
    ax.set_ylabel("Deduplicated Fitness Score", fontsize=10)
    ax.set_title("(a) Conformance Fitness", fontsize=11, fontweight="bold")
    ax.set_xlim(0.13, 0.62)
    ax.grid(True, alpha=0.3)

    # Panel 2: Shadow %
    ax = axes[1]
    q25 = [np.percentile(s, 25) for s in shadow_all]
    q75 = [np.percentile(s, 75) for s in shadow_all]
    ax.fill_between(thresholds, q25, q75, alpha=0.2, color=colors["shadow"])
    ax.plot(thresholds, mean_shadow, "o-", color=colors["shadow"], linewidth=2, markersize=5)
    ax.set_xlabel("SBERT Cosine Similarity Threshold", fontsize=10)
    ax.set_ylabel("Shadow Activity Percentage (%)", fontsize=10)
    ax.set_title("(b) Shadow Prevalence", fontsize=11, fontweight="bold")
    ax.set_xlim(0.13, 0.62)
    ax.grid(True, alpha=0.3)

    # Panel 3: Agenda Coverage
    ax = axes[2]
    q25 = [np.percentile(c, 25) for c in coverage_all]
    q75 = [np.percentile(c, 75) for c in coverage_all]
    ax.fill_between(thresholds, q25, q75, alpha=0.2, color=colors["coverage"])
    ax.plot(thresholds, mean_coverage, "o-", color=colors["coverage"], linewidth=2, markersize=5)
    ax.set_xlabel("SBERT Cosine Similarity Threshold", fontsize=10)
    ax.set_ylabel("Agenda Coverage (%)", fontsize=10)
    ax.set_title("(c) Agenda Coverage", fontsize=11, fontweight="bold")
    ax.set_xlim(0.13, 0.62)
    ax.grid(True, alpha=0.3)

    # Mark the operating point (0.35)
    for ax in axes:
        ax.axvline(x=0.35, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.annotate("t=0.35\n(used)", xy=(0.35, ax.get_ylim()[1]),
                    xytext=(0.35, ax.get_ylim()[1] * 0.92),
                    fontsize=8, ha="center", color="gray")

    fig.suptitle("SBERT Threshold Sensitivity Analysis Across 54 Meetings",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig_path = os.path.join(output_dir, "sbert_sensitivity.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
