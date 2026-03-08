#!/usr/bin/env python3
"""Generate thesis-quality figures from batch meeting analysis results.

Figures are organized around the thesis narrative:
  1. The METHOD: multimodal extraction pipeline effectiveness
  2. The PHENOMENON: shadow workflows and conformance gaps
  3. The PATTERNS: structural and declarative analysis findings

NOT organized by city comparison (cities are the dataset, not the research question).

Usage:
    python generate_figures.py --meetings-dir meetings/ --output-dir thesis/figures/
    python generate_figures.py --meetings-dir meetings/ --output-dir thesis/figures/ --format pdf --dpi 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================

CITY_COLORS = {
    "Seattle": "#2196F3",
    "Denver": "#FF9800",
    "Boston": "#4CAF50",
    "Alameda": "#9C27B0",
}

CITY_MARKERS = {
    "Seattle": "o",
    "Denver": "s",
    "Boston": "^",
    "Alameda": "D",
}

DEVIANCE_COLORS = {
    "benign": "#66BB6A",
    "violation": "#EF5350",
    "efficiency": "#42A5F5",
    "innovation": "#FFA726",
    "disruption": "#BDBDBD",
    "unknown": "#E0E0E0",
}

SOURCE_COLORS = {
    "audio": "#1976D2",
    "visual": "#E64A19",
    "fused": "#7B1FA2",
}


def get_color(city: str) -> str:
    return CITY_COLORS.get(city, "#607D8B")


def get_marker(city: str) -> str:
    return CITY_MARKERS.get(city, "o")


# ============================================================
# Data Loading
# ============================================================

def load_metrics(metrics_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    return df


def load_all_conformance(meetings_dir: str) -> list[dict]:
    results = []
    for folder in sorted(os.listdir(meetings_dir)):
        conf_path = os.path.join(meetings_dir, folder, "conformance.json")
        if os.path.exists(conf_path):
            with open(conf_path) as f:
                results.append(json.load(f))
    return results


# ============================================================
# Thesis-quality plot defaults
# ============================================================

def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (7, 4.5),
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ============================================================
# FIGURE 1: Corpus Overview Table-Figure
# Story: "We analyzed 54 meetings across 4 cities — here's the dataset"
# ============================================================

def fig_corpus_overview(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")

    cities = sorted(df["city"].unique())
    table_data = []
    for city in cities:
        cdf = df[df["city"] == city]
        n = len(cdf)
        hours = cdf["duration_seconds"].sum() / 3600
        events = cdf["raw_events"].sum()
        table_data.append([city, str(n), f"{hours:.1f}", f"{events:,}"])

    # Totals
    table_data.append([
        "Total", str(len(df)),
        f"{df['duration_seconds'].sum()/3600:.1f}",
        f"{df['raw_events'].sum():,}"
    ])

    col_labels = ["City", "Meetings", "Hours", "Raw Events"]
    table = ax.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Style total row
    for j in range(len(col_labels)):
        table[len(table_data), j].set_facecolor("#ECEFF1")
        table[len(table_data), j].set_text_props(fontweight="bold")
    # City colors
    for i, city in enumerate(cities):
        table[i + 1, 0].set_text_props(color=get_color(city), fontweight="bold")

    ax.set_title("Meeting Corpus Overview", fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"corpus_overview.{fmt}"), dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# FIGURE 2: Modality Contribution
# Story: "Audio dominates (57%), but multimodal fusion adds 39%"
# ============================================================

def fig_modality_contribution(conformance_list: list[dict], output_dir: str, fmt: str, dpi: int):
    totals = {"audio": 0, "visual": 0, "fused": 0}
    for c in conformance_list:
        sources = c.get("extraction", {}).get("sources", {})
        totals["audio"] += sources.get("audio", 0)
        totals["visual"] += sources.get("visual", 0)
        totals["fused"] += sources.get("fused", 0)

    total = sum(totals.values())
    labels = ["Audio\n(Whisper)", "Visual\n(RTMPose)", "Multimodal\nFusion"]
    values = [totals["audio"], totals["visual"], totals["fused"]]
    colors = [SOURCE_COLORS["audio"], SOURCE_COLORS["visual"], SOURCE_COLORS["fused"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: donut chart
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, colors=colors, autopct=lambda p: f"{p:.1f}%\n({int(p*total/100):,})",
        startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.45),
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax1.set_title("Event Source Distribution", fontsize=12)

    # Right: per-meeting box plot of source ratios
    audio_pcts, visual_pcts, fused_pcts = [], [], []
    for c in conformance_list:
        sources = c.get("extraction", {}).get("sources", {})
        t = sources.get("audio", 0) + sources.get("visual", 0) + sources.get("fused", 0)
        if t > 0:
            audio_pcts.append(sources.get("audio", 0) / t * 100)
            visual_pcts.append(sources.get("visual", 0) / t * 100)
            fused_pcts.append(sources.get("fused", 0) / t * 100)

    bp = ax2.boxplot([audio_pcts, visual_pcts, fused_pcts],
                     labels=["Audio", "Visual", "Fusion"],
                     patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # Jitter
    for i, (vals, color) in enumerate(zip([audio_pcts, visual_pcts, fused_pcts], colors)):
        jitter = np.random.normal(0, 0.05, len(vals))
        ax2.scatter(np.full(len(vals), i + 1) + jitter, vals,
                    c=color, alpha=0.5, s=15, zorder=3)

    ax2.set_ylabel("Percentage of Events per Meeting (%)")
    ax2.set_title("Per-Meeting Modality Variation", fontsize=12)

    fig.suptitle("Multimodal Event Extraction: Source Modality Contribution",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"modality_contribution.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 3: Fitness Distribution (histogram)
# Story: "Most meetings have low conformance — median fitness 0.286"
# ============================================================

def fig_fitness_distribution(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    scores = df["fitness_dedup"].dropna().values
    ax.hist(scores, bins=12, color="#1976D2", alpha=0.7, edgecolor="white", linewidth=0.5)

    mean_val = np.mean(scores)
    median_val = np.median(scores)
    ax.axvline(mean_val, color="#E53935", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_val:.3f}")
    ax.axvline(median_val, color="#FF8F00", linestyle="-.", linewidth=1.5,
               label=f"Median: {median_val:.3f}")

    ax.set_xlabel("Token-Replay Fitness (Deduplicated)")
    ax.set_ylabel("Number of Meetings")
    ax.set_title("Process Conformance: Fitness Score Distribution (n=54)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fitness_distribution.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 4: Raw vs Dedup Fitness
# Story: "Deduplication systematically improves fitness measurement"
# ============================================================

def fig_fitness_raw_vs_dedup(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(6, 6))

    for city in sorted(df["city"].unique()):
        mask = df["city"] == city
        ax.scatter(df[mask]["fitness_raw"], df[mask]["fitness_dedup"],
                   c=get_color(city), marker=get_marker(city),
                   label=city, alpha=0.7, s=45, zorder=3)

    lim = max(df["fitness_raw"].max(), df["fitness_dedup"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.2, label="No improvement")
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.04, color="green")
    ax.text(0.05, lim * 0.92, "Dedup improves fitness", fontsize=8,
            color="green", alpha=0.6, style="italic")

    ax.set_xlabel("Raw Fitness Score")
    ax.set_ylabel("Deduplicated Fitness Score")
    ax.set_title("Effect of Deduplication on Fitness Measurement")
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fitness_raw_vs_dedup.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 5: Shadow vs Formal Split
# Story: "54.6% of detected activities are shadow (off-agenda)"
# ============================================================

def fig_shadow_formal_split(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    gridspec_kw={"width_ratios": [1, 1.5]})

    total_formal = df["formal_events"].sum()
    total_shadow = df["shadow_events"].sum()

    # Left: overall split donut
    ax1.pie([total_formal, total_shadow],
            labels=[f"Formal\n({total_formal:,})", f"Shadow\n({total_shadow:,})"],
            colors=["#43A047", "#E53935"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.45), textprops={"fontsize": 10})
    ax1.set_title("Overall Activity Split", fontsize=12)

    # Right: histogram of shadow % across meetings
    shadow_pcts = df["shadow_pct"].values * 100
    ax2.hist(shadow_pcts, bins=12, color="#E53935", alpha=0.6, edgecolor="white")
    mean_s = np.mean(shadow_pcts)
    median_s = np.median(shadow_pcts)
    ax2.axvline(mean_s, color="#B71C1C", linestyle="--", linewidth=1.5,
                label=f"Mean: {mean_s:.1f}%")
    ax2.axvline(median_s, color="#FF8F00", linestyle="-.", linewidth=1.5,
                label=f"Median: {median_s:.1f}%")
    ax2.set_xlabel("Shadow Activity Percentage (%)")
    ax2.set_ylabel("Number of Meetings")
    ax2.set_title("Shadow Prevalence Distribution", fontsize=12)
    ax2.legend()

    fig.suptitle("Shadow Workflow Prevalence Across 54 Meetings",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"shadow_formal_split.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 6: Deviance Taxonomy
# Story: "Shadow activities are mostly innovative — substantive off-agenda discussions"
# ============================================================

def fig_deviance_taxonomy(conformance_list: list[dict], output_dir: str, fmt: str, dpi: int):
    totals = {"benign": 0, "innovation": 0, "efficiency": 0,
              "disruption": 0, "unknown": 0}
    for c in conformance_list:
        dev = c.get("deviance", {})
        for k in totals:
            totals[k] += dev.get(k, 0)

    # Remove zero categories
    labels = [k.title() for k, v in totals.items() if v > 0]
    values = [v for v in totals.values() if v > 0]
    colors = [DEVIANCE_COLORS[k] for k, v in totals.items() if v > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: donut
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(p*sum(values)/100):,})",
        startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.4),
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax1.set_title("Aggregate Deviance\nClassification", fontsize=11)

    # Right: horizontal bar with annotations
    ax2.barh(range(len(labels)), values, color=colors, alpha=0.85)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Number of Shadow Events")
    ax2.set_title("Deviance Category Counts", fontsize=11)
    ax2.invert_yaxis()
    for i, v in enumerate(values):
        ax2.text(v + max(values) * 0.02, i, f"{v:,}", va="center", fontsize=9)

    fig.suptitle("Process Deviance Classification of Shadow Activities",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"deviance_taxonomy.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 7: POWL Shadow Patterns
# Story: "Isolated and concurrent patterns dominate shadow structure"
# ============================================================

def fig_powl_patterns(conformance_list: list[dict], output_dir: str, fmt: str, dpi: int):
    totals = {"isolated": 0, "concurrent": 0, "sequential": 0, "recurring": 0}
    for c in conformance_list:
        types = c.get("powl", {}).get("cluster_types", {})
        for k in totals:
            totals[k] += types.get(k, 0)

    labels_map = {
        "isolated": "Isolated\n(one-off shadow activities)",
        "concurrent": "Concurrent\n(parallel shadow threads)",
        "sequential": "Sequential\n(ordered shadow chains)",
        "recurring": "Recurring\n(repeated shadow patterns)",
    }

    labels = [labels_map[k] for k, v in totals.items() if v > 0]
    values = [v for v in totals.values() if v > 0]
    colors = ["#78909C", "#FF7043", "#66BB6A", "#42A5F5"][:len(labels)]

    if not values:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(p*sum(values)/100)})",
        startangle=90, pctdistance=0.78, wedgeprops=dict(width=0.42),
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("POWL Shadow Workflow Structural Patterns (n=1,065 clusters)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"powl_patterns.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 8: Declare Compliance (bimodal histogram)
# Story: "Compliance is bimodal — meetings are either highly
#         compliant or have zero Declare constraint coverage"
# ============================================================

def fig_declare_compliance(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Grade bands
    bands = [(90, 100, "A", "#43A047"), (80, 90, "B", "#7CB342"),
             (70, 80, "C", "#FDD835"), (60, 70, "D", "#FF8F00"), (0, 60, "F", "#E53935")]
    for lo, hi, grade, color in bands:
        ax.axvspan(lo, hi, alpha=0.06, color=color)

    scores = df["declare_score"].dropna().values
    ax.hist(scores, bins=20, color="#1976D2", alpha=0.7, edgecolor="white")

    # Grade annotations at top
    for lo, hi, grade, color in bands:
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1,
                grade, ha="center", va="top", fontsize=16, color=color, alpha=0.4,
                fontweight="bold")

    mean_val = np.mean(scores)
    ax.axvline(mean_val, color="#E53935", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_val:.1f}")

    # Add grade count annotation
    grade_counts = Counter(df["declare_grade"].values)
    grade_text = "  ".join([f"{g}: {grade_counts.get(g, 0)}" for g in ["A", "B", "C", "D", "F"]])
    ax.text(0.98, 0.95, grade_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Robert's Rules Compliance Score (0–100)")
    ax.set_ylabel("Number of Meetings")
    ax.set_title("Declare Constraint Compliance Distribution (n=54)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"declare_compliance.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 9: Declare Violations Detail
# Story: "Which Robert's Rules constraints are most violated?"
# ============================================================

def fig_declare_violations(conformance_list: list[dict], output_dir: str, fmt: str, dpi: int):
    # Parse violation details
    violation_counts = Counter()
    for c in conformance_list:
        violations = c.get("declare", {}).get("violations", [])
        for v in violations:
            if isinstance(v, dict):
                tmpl = v.get("template", "")
                a = v.get("a", "")
                b = v.get("b", "")
                severity = v.get("severity", "")
                if b:
                    name = f"{tmpl}({a}, {b})"
                else:
                    name = f"{tmpl}({a})"
                violation_counts[name] += 1

    # Also show satisfied vs violated aggregate
    total_sat = sum(c.get("declare", {}).get("satisfied", 0) for c in conformance_list)
    total_viol = sum(c.get("declare", {}).get("violated", 0) for c in conformance_list)
    meetings_with_constraints = sum(
        1 for c in conformance_list
        if c.get("declare", {}).get("constraints_checked", 0) > 0
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: satisfied vs violated donut
    if total_sat + total_viol > 0:
        ax1.pie([total_sat, total_viol],
                labels=[f"Satisfied\n({total_sat})", f"Violated\n({total_viol})"],
                colors=["#43A047", "#E53935"],
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(width=0.45), textprops={"fontsize": 10})
    ax1.set_title(f"Constraint Outcomes\n({meetings_with_constraints} meetings with constraints)",
                  fontsize=10)

    # Right: specific violations bar
    if violation_counts:
        sorted_v = violation_counts.most_common(10)
        names = [v[0] for v in sorted_v]
        counts = [v[1] for v in sorted_v]

        y = np.arange(len(names))
        ax2.barh(y, counts, color="#E53935", alpha=0.8)
        ax2.set_yticks(y)
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel("Number of Meetings")
        ax2.invert_yaxis()
        for i, v in enumerate(counts):
            ax2.text(v + 0.2, i, str(v), va="center", fontsize=9)
    else:
        ax2.text(0.5, 0.5, f"No individual violations\nrecorded in data\n\n"
                 f"{total_viol} total constraint\nviolations across\n{meetings_with_constraints} meetings",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=10, style="italic", color="#757575")
        ax2.set_axis_off()

    ax2.set_title("Most Violated Constraints", fontsize=10)

    fig.suptitle("Robert's Rules Declare Constraint Analysis",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"declare_violations.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 10: Fitness vs Shadow Correlation
# Story: "Is there a relationship between shadow prevalence and conformance?"
# ============================================================

def fig_fitness_vs_shadow(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 5))

    for city in sorted(df["city"].unique()):
        mask = df["city"] == city
        ax.scatter(df[mask]["shadow_pct"] * 100, df[mask]["fitness_dedup"],
                   c=get_color(city), marker=get_marker(city),
                   label=city, alpha=0.7, s=45, zorder=3)

    # Trend line
    x = df["shadow_pct"].values * 100
    y = df["fitness_dedup"].values
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() > 2:
        z = np.polyfit(x[valid], y[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=1.5,
                label=f"Linear trend (slope={z[0]:.4f})")

        # R-squared
        y_pred = p(x[valid])
        ss_res = np.sum((y[valid] - y_pred) ** 2)
        ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.02, 0.02, f"R² = {r2:.3f}", transform=ax.transAxes,
                fontsize=9, style="italic", color="#757575")

    ax.set_xlabel("Shadow Activity Percentage (%)")
    ax.set_ylabel("Deduplicated Fitness Score")
    ax.set_title("Conformance Fitness vs. Shadow Workflow Prevalence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"fitness_vs_shadow.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 11: Agenda Coverage Distribution
# Story: "The pipeline detects ~47% of agenda items on average"
# ============================================================

def fig_agenda_coverage(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    coverages = df["agenda_coverage_pct"].dropna().values * 100
    ax.hist(coverages, bins=12, color="#1976D2", alpha=0.7, edgecolor="white")

    mean_c = np.mean(coverages)
    median_c = np.median(coverages)
    ax.axvline(mean_c, color="#E53935", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_c:.1f}%")
    ax.axvline(median_c, color="#FF8F00", linestyle="-.", linewidth=1.5,
               label=f"Median: {median_c:.1f}%")

    ax.set_xlabel("Agenda Item Coverage (%)")
    ax.set_ylabel("Number of Meetings")
    ax.set_title("Agenda Item Detection Rate Distribution (n=54)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"agenda_coverage.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# FIGURE 12: Conformance Summary (multi-metric scatter)
# Story: "Each meeting's governance profile at a glance"
# ============================================================

def fig_conformance_scatter(df: pd.DataFrame, output_dir: str, fmt: str, dpi: int):
    fig, ax = plt.subplots(figsize=(8, 5))

    for city in sorted(df["city"].unique()):
        mask = df["city"] == city
        cdf = df[mask]
        sizes = cdf["agenda_coverage_pct"] * 200 + 20  # Size = coverage
        ax.scatter(cdf["fitness_dedup"], cdf["declare_score"],
                   c=get_color(city), marker=get_marker(city),
                   s=sizes, label=city, alpha=0.6, zorder=3,
                   edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Deduplicated Fitness Score")
    ax.set_ylabel("Declare Compliance Score (0–100)")
    ax.set_title("Meeting Governance Profile: Fitness × Compliance × Coverage")
    ax.legend(fontsize=8)

    # Size legend
    for sz, label in [(0.2, "20%"), (0.5, "50%"), (1.0, "100%")]:
        ax.scatter([], [], c="gray", alpha=0.4, s=sz * 200 + 20,
                   label=f"Coverage: {label}")
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"conformance_scatter.{fmt}"), dpi=dpi)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis figures from batch analysis results",
    )
    parser.add_argument("--meetings-dir", default="meetings",
                        help="Path to meetings/ folder")
    parser.add_argument("--output-dir", default="thesis/figures",
                        help="Output directory for figures")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Figure output format")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    metrics_csv = os.path.join(args.meetings_dir, "_aggregated", "metrics.csv")
    if not os.path.exists(metrics_csv):
        print(f"ERROR: {metrics_csv} not found. Run batch_analyze.py first.")
        sys.exit(1)

    df = load_metrics(metrics_csv)
    conformance_list = load_all_conformance(args.meetings_dir)

    print(f"Loaded {len(df)} meetings from {metrics_csv}")
    print(f"Generating figures to {args.output_dir}/\n")

    figures = [
        # METHOD figures
        ("corpus_overview", fig_corpus_overview, [df]),
        ("modality_contribution", fig_modality_contribution, [conformance_list]),

        # CONFORMANCE figures
        ("fitness_distribution", fig_fitness_distribution, [df]),
        ("fitness_raw_vs_dedup", fig_fitness_raw_vs_dedup, [df]),
        ("agenda_coverage", fig_agenda_coverage, [df]),

        # SHADOW WORKFLOW figures
        ("shadow_formal_split", fig_shadow_formal_split, [df]),
        ("deviance_taxonomy", fig_deviance_taxonomy, [conformance_list]),
        ("powl_patterns", fig_powl_patterns, [conformance_list]),
        ("fitness_vs_shadow", fig_fitness_vs_shadow, [df]),

        # DECLARE figures
        ("declare_compliance", fig_declare_compliance, [df]),
        ("declare_violations", fig_declare_violations, [conformance_list]),

        # SYNTHESIS figure
        ("conformance_scatter", fig_conformance_scatter, [df]),
    ]

    generated = 0
    total = len(figures)
    for i, (name, func, extra_args) in enumerate(figures, 1):
        try:
            func(*extra_args, args.output_dir, args.format, args.dpi)
            generated += 1
            print(f"  [{i}/{total}] {name}.{args.format}")
        except Exception as e:
            print(f"  [{i}/{total}] ERROR {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone: {generated}/{total} figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
