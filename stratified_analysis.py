#!/usr/bin/env python3
"""Stratified analysis: fitness by agenda complexity + shadow activity examples."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

MEETINGS_DIR = "meetings"
METRICS_CSV = os.path.join(MEETINGS_DIR, "_aggregated", "metrics.csv")
FIGURES_DIR = os.path.join("thesis", "figures")

# Agenda complexity bins
BINS = [
    ("Sparse ($\\leq 5$)", 0, 5),
    ("Moderate (6--50)", 6, 50),
    ("Dense (51--100)", 51, 100),
]


def load_metrics():
    df = pd.read_csv(METRICS_CSV)
    return df


def assign_tier(n_items):
    for label, lo, hi in BINS:
        if lo <= n_items <= hi:
            return label
    return "Unknown"


def stratified_table(df):
    df["tier"] = df["agenda_items"].apply(assign_tier)

    # Order tiers
    tier_order = [b[0] for b in BINS]
    df["tier"] = pd.Categorical(df["tier"], categories=tier_order, ordered=True)

    grouped = df.groupby("tier", observed=True)

    rows = []
    for tier in tier_order:
        g = grouped.get_group(tier)
        rows.append({
            "Tier": tier,
            "N": len(g),
            "Mean Items": f"{g['agenda_items'].mean():.1f}",
            "Mean Fitness": f"{g['fitness_dedup'].mean():.3f}",
            "Median Fitness": f"{g['fitness_dedup'].median():.3f}",
            "Mean Shadow %": f"{g['shadow_pct'].mean() * 100:.1f}",
            "Mean Coverage %": f"{g['agenda_coverage_pct'].mean() * 100:.1f}",
        })

    result = pd.DataFrame(rows)
    print("\n=== Stratified Analysis: Fitness by Agenda Complexity ===\n")
    print(result.to_string(index=False))

    # LaTeX output
    print("\n=== LaTeX Table ===\n")
    print(r"\begin{table}[htbp]")
    print(r"    \centering")
    print(r"    \caption{Fitness and shadow prevalence stratified by agenda complexity. Meetings with sparse agendas achieve higher fitness but higher shadow prevalence, while dense agendas yield lower fitness but lower shadow prevalence.}")
    print(r"    \label{tab:stratified}")
    print(r"    \small")
    print(r"    \begin{tabular}{lccccc}")
    print(r"        \toprule")
    print(r"        Agenda Tier & $N$ & Mean Items & Fitness (dedup) & Shadow~\% & Coverage~\% \\")
    print(r"        \midrule")
    for r in rows:
        print(f"        {r['Tier']} & {r['N']} & {r['Mean Items']} & {r['Mean Fitness']} & {r['Mean Shadow %']} & {r['Mean Coverage %']} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")

    return df


def box_plot(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    tier_order = [b[0] for b in BINS]
    # Clean labels for plot (remove LaTeX)
    plot_labels = ["Sparse\n(≤5 items)", "Moderate\n(6–50 items)", "Dense\n(51–100 items)"]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    for ax_idx, (metric, ylabel, scale) in enumerate([
        ("fitness_dedup", "Deduplicated Fitness", 1),
        ("shadow_pct", "Shadow Activity (%)", 100),
        ("agenda_coverage_pct", "Agenda Coverage (%)", 100),
    ]):
        ax = axes[ax_idx]
        data = []
        for tier in tier_order:
            vals = df[df["tier"] == tier][metric].values * scale
            data.append(vals)

        bp = ax.boxplot(data, labels=plot_labels, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Meeting Metrics Stratified by Agenda Complexity", fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fitness_by_agenda_complexity.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: {out_path}")
    plt.close()


def find_shadow_examples():
    """Scan mapped_events.csv across all meetings for diverse shadow examples."""
    print("\n=== Shadow Activity Examples ===\n")

    examples = []
    for folder in sorted(os.listdir(MEETINGS_DIR)):
        mapped_path = os.path.join(MEETINGS_DIR, folder, "variant_llm", "mapped_events.csv")
        if not os.path.isfile(mapped_path):
            continue

        try:
            events = pd.read_csv(mapped_path)
        except Exception:
            continue

        # Find shadow activities
        label_col = None
        for col in ["mapped_activity", "activity", "abstracted_activity"]:
            if col in events.columns:
                label_col = col
                break
        if label_col is None:
            continue

        shadow_mask = events[label_col].str.contains("Shadow:", case=False, na=False)
        shadow_events = events[shadow_mask]

        for _, row in shadow_events.iterrows():
            label = row[label_col]
            # Extract topic from "Shadow: Unscheduled discussion of X" or "Shadow: Off-agenda discussion about X"
            if "discussion" in label.lower() or "debate" in label.lower():
                city = folder.split("_")[0].capitalize()
                date = "_".join(folder.split("_")[1:])
                examples.append({
                    "city": city,
                    "meeting": folder,
                    "date": date,
                    "label": label,
                    "timestamp": row.get("timestamp", ""),
                })

    # Deduplicate by label content and pick diverse examples
    seen_topics = set()
    diverse = []
    # Prioritize diversity: different cities, different topics
    for ex in examples:
        topic_key = ex["label"].lower()
        # Skip generic ones
        if topic_key in seen_topics:
            continue
        seen_topics.add(topic_key)
        diverse.append(ex)

    # Show top examples grouped by city
    by_city = {}
    for ex in diverse:
        by_city.setdefault(ex["city"], []).append(ex)

    for city in sorted(by_city.keys()):
        print(f"\n{city}:")
        for ex in by_city[city][:5]:
            print(f"  [{ex['date']}] {ex['label']}")

    return diverse


if __name__ == "__main__":
    df = load_metrics()
    df = stratified_table(df)
    box_plot(df)
    find_shadow_examples()
