#!/usr/bin/env python3
"""Batch analysis of downloaded city council meetings.

Runs the full Meeting Process Twin pipeline on all meetings in the
meetings/ directory, producing structured conformance results per the
layered data schema.

Usage:
    # Dry run — validate inputs, report layer status
    python batch_analyze.py --dry-run --api-key sk-...

    # Process a single meeting
    python batch_analyze.py --only-meeting seattle_2026-01-20 --api-key sk-...

    # Process all meetings (Layer 2 is expensive — uses local Whisper)
    python batch_analyze.py --api-key sk-...

    # Skip video processing if raw_events.csv exists
    python batch_analyze.py --api-key sk-... --skip-video

    # Only specific cities
    python batch_analyze.py --api-key sk-... --only-cities seattle,denver
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import time
import traceback
from datetime import datetime, date
from collections import Counter
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock Streamlit BEFORE importing any project module
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
from bpmn_gen import generate_agenda_bpmn, convert_to_event_log
from pipeline.time_utils import ts_to_seconds

# Research modules
from research.powl.powl_discovery import discover_shadow_powl, discover_full_powl
from research.powl.powl_analysis import get_powl_statistics
from research.powl.shadow_patterns import (
    detect_shadow_clusters,
    classify_shadow_patterns,
    get_pattern_summary,
)
from research.declare.declare_conformance import (
    check_roberts_rules_conformance,
    compute_procedural_compliance_score,
)
from research.declare.violation_analysis import classify_violations, get_violation_summary
from research.deviance.deviance_classifier import DevianceClassifier


# ============================================================
# Utility helpers
# ============================================================

def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


# ============================================================
# Meeting discovery and validation
# ============================================================

def discover_meetings(
    meetings_dir: str,
    only_cities: list[str] | None = None,
    only_meeting: str | None = None,
) -> list[dict]:
    """Scan meetings/ directory for valid meeting folders."""
    meetings = []

    for folder_name in sorted(os.listdir(meetings_dir)):
        folder_path = os.path.join(meetings_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name.startswith("_"):
            continue

        # Parse city and date from folder name
        parts = folder_name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        city_key, date_str = parts

        # Apply filters
        if only_meeting and folder_name != only_meeting:
            continue
        if only_cities and city_key not in only_cities:
            continue

        # Load metadata
        metadata_path = os.path.join(folder_path, "metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        meetings.append({
            "folder_name": folder_name,
            "path": folder_path,
            "city_key": city_key,
            "city": metadata.get("city", city_key.title()),
            "date": metadata.get("date", date_str),
            "has_video": os.path.exists(os.path.join(folder_path, "video.mp4")),
            "has_agenda": os.path.exists(os.path.join(folder_path, "agenda.txt")),
            "has_metadata": bool(metadata),
            "metadata": metadata,
        })

    return meetings


def check_layer_status(meeting_path: str) -> dict:
    """Determine which processing layers are already complete."""
    raw_csv = os.path.join(meeting_path, "raw_events.csv")
    variant_dir = os.path.join(meeting_path, "variant_llm")
    mapped_csv = os.path.join(variant_dir, "mapped_events.csv")
    conformance_json = os.path.join(meeting_path, "conformance.json")

    layer2 = os.path.exists(raw_csv) and os.path.getsize(raw_csv) > 100
    layer3 = os.path.exists(mapped_csv) and os.path.getsize(mapped_csv) > 100
    layer4 = os.path.exists(conformance_json)

    raw_count = None
    if layer2:
        try:
            df = pd.read_csv(raw_csv)
            raw_count = len(df)
        except Exception:
            pass

    return {
        "layer2_done": layer2,
        "layer3_done": layer3,
        "layer4_done": layer4,
        "raw_event_count": raw_count,
    }


# ============================================================
# Layer 2: Video Processing (Whisper + NLP + RTMPose)
# ============================================================

def compute_source_distribution(df_raw: pd.DataFrame) -> dict:
    """Compute event source breakdown from raw events."""
    if "source" not in df_raw.columns:
        return {"audio": len(df_raw), "visual": 0, "nlp": 0, "fused": 0, "total": len(df_raw)}

    source_counts = df_raw["source"].str.lower().value_counts()
    dist = {
        "audio": int(source_counts.get("audio", 0)),
        "visual": int(source_counts.get("video", 0)),
        "nlp": int(source_counts.get("nlp", 0)),
        "fused": int(source_counts.get("fused (audio+video)", 0)
                     + source_counts.get("audio+motion", 0)),
        "total": len(df_raw),
    }
    return dist


def run_layer2(
    meeting: dict,
    api_key: str,
    use_local_whisper: bool = True,
    whisper_model: str = "small",
    debug: bool = False,
) -> dict:
    """Process video -> raw_events.csv + extraction_config.json + source_distribution.json."""
    video_path = os.path.join(meeting["path"], "video.mp4")
    t0 = time.time()

    processor = VideoProcessor(api_key or "local", debug=debug)
    df_raw = processor.process_video(
        video_path,
        use_local_whisper=use_local_whisper,
        local_whisper_model=whisper_model,
    )

    # Free GPU memory after each video to prevent CUDA OOM on subsequent meetings
    del processor
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc; gc.collect()
    except Exception:
        pass

    elapsed = time.time() - t0

    # Save raw_events.csv
    raw_csv = os.path.join(meeting["path"], "raw_events.csv")
    df_raw.to_csv(raw_csv, index=False)

    # Source distribution
    source_dist = compute_source_distribution(df_raw)
    with open(os.path.join(meeting["path"], "source_distribution.json"), "w") as f:
        json.dump(source_dist, f, indent=2)

    # Extraction config
    config = {
        "whisper_model": whisper_model,
        "use_local_whisper": use_local_whisper,
        "raw_event_count": len(df_raw),
        "processing_time_seconds": round(elapsed, 1),
        "processed_at": datetime.now().isoformat(),
    }
    with open(os.path.join(meeting["path"], "extraction_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Estimate duration from last event timestamp
    duration_s = 0
    if "timestamp" in df_raw.columns and len(df_raw) > 0:
        try:
            duration_s = int(ts_to_seconds(str(df_raw["timestamp"].iloc[-1])))
        except Exception:
            pass

    print(f"    Layer 2: {len(df_raw)} raw events extracted in {elapsed:.0f}s")
    return {
        "df_raw": df_raw,
        "raw_event_count": len(df_raw),
        "source_distribution": source_dist,
        "duration_seconds": duration_s,
        "elapsed_seconds": elapsed,
    }


# ============================================================
# Layer 3: LLM Abstraction + SBERT Mapping
# ============================================================

def run_layer3(
    meeting: dict,
    df_raw: pd.DataFrame,
    agenda_text: str,
    api_key: str,
    params: dict,
    engine: ComplianceEngine,
    debug: bool = False,
) -> dict:
    """LLM abstraction + SBERT mapping -> variant_llm/ folder."""
    t0 = time.time()

    # Generate agenda BPMN
    _, activities, bpmn_obj = generate_agenda_bpmn(agenda_text, api_key)
    print(f"    Agenda BPMN: {len(activities)} activities")

    # LLM abstraction
    df_abstracted = engine.abstract_events_df(
        df=df_raw,
        agenda_tasks=activities,
        window_seconds=params["window_seconds"],
        overlap_ratio=params["overlap_ratio"],
        min_events_per_window=params["min_events"],
        min_label_support=params["min_label_support"],
        shadow_min_ratio=params.get("shadow_min_ratio", 0.15),
        api_key=api_key,
        openai_model=params["openai_model"],
        openai_timeout=params.get("openai_timeout", 60),
        max_windows_per_run=params.get("max_windows", 150),
        cache={},
    )
    print(f"    Abstraction: {len(df_abstracted)} events")

    # SBERT mapping
    df_mapped = engine.map_events_to_agenda(
        df_abstracted, activities, threshold=params["sbert_threshold"],
    )

    # Count formal vs shadow
    if "mapped_activity" in df_mapped.columns:
        shadow_mask = df_mapped["mapped_activity"].str.startswith("Deviation:", na=False)
        formal_count = int((~shadow_mask).sum())
        shadow_count = int(shadow_mask.sum())
    else:
        formal_count = len(df_mapped)
        shadow_count = 0

    print(f"    Mapped: {formal_count} formal, {shadow_count} shadow")

    # Save to variant_llm/
    variant_dir = os.path.join(meeting["path"], "variant_llm")
    os.makedirs(variant_dir, exist_ok=True)

    df_abstracted.to_csv(os.path.join(variant_dir, "abstracted_events.csv"), index=False)
    df_mapped.to_csv(os.path.join(variant_dir, "mapped_events.csv"), index=False)

    with open(os.path.join(variant_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    elapsed = time.time() - t0
    print(f"    Layer 3: completed in {elapsed:.0f}s")

    return {
        "activities": activities,
        "bpmn_obj": bpmn_obj,
        "df_abstracted": df_abstracted,
        "df_mapped": df_mapped,
        "formal_count": formal_count,
        "shadow_count": shadow_count,
        "elapsed_seconds": elapsed,
    }


# ============================================================
# Layer 4: Conformance Analysis
# ============================================================

def _run_with_timeout(func, timeout_sec=120):
    """Run a function with a timeout. Returns (result, timed_out)."""
    import threading
    result_holder = [None]
    error_holder = [None]

    def target():
        try:
            result_holder[0] = func()
        except Exception as e:
            error_holder[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        return None, True  # timed out
    if error_holder[0]:
        raise error_holder[0]
    return result_holder[0], False


def compute_bpmn_fitness(
    df_mapped: pd.DataFrame,
    bpmn_obj,
    activities: list[str],
    engine: ComplianceEngine,
) -> dict:
    """Compute raw and dedup BPMN fitness scores."""
    result = {
        "raw": 0.0, "dedup": 0.0,
        "match_rate": 0.0, "agenda_coverage": 0, "agenda_coverage_pct": 0.0,
    }

    if df_mapped.empty or bpmn_obj is None:
        return result

    act_col = "mapped_activity" if "mapped_activity" in df_mapped.columns else "activity_name"

    # Match rate and agenda coverage
    shadow_mask = df_mapped[act_col].str.startswith("Deviation:", na=False)
    matched = df_mapped[~shadow_mask]
    result["match_rate"] = round(len(matched) / len(df_mapped), 4) if len(df_mapped) > 0 else 0.0

    covered_items = set(matched[act_col].unique()) if not matched.empty else set()
    result["agenda_coverage"] = len(covered_items)
    result["agenda_coverage_pct"] = round(
        len(covered_items) / len(activities), 4
    ) if activities else 0.0

    # Raw fitness (with 120s timeout)
    log_data = convert_to_event_log(df_mapped)
    if log_data is not None and not log_data.empty:
        try:
            fit_val, timed_out = _run_with_timeout(
                lambda: engine.calculate_fitness(bpmn_obj, log_data), 120
            )
            if timed_out:
                print(f"    WARNING: Raw fitness timed out (120s) — skipping")
            elif fit_val is not None:
                result["raw"] = round(fit_val.get("score", 0.0), 4)
        except Exception as e:
            print(f"    WARNING: Raw fitness error: {e}")

    # Dedup fitness (first occurrence per agenda item, with 120s timeout)
    try:
        formal = df_mapped[~shadow_mask].copy()
        if not formal.empty:
            formal["__ts"] = formal["timestamp"].apply(ts_to_seconds)
            formal = formal.sort_values("__ts")
            first = formal.groupby(act_col).first().reset_index()
            first = first.sort_values("__ts").drop(columns=["__ts"])
            first["activity_name"] = first[act_col]
            dedup_log = convert_to_event_log(first)
            if dedup_log is not None and not dedup_log.empty:
                dedup_val, timed_out = _run_with_timeout(
                    lambda: engine.calculate_fitness(bpmn_obj, dedup_log), 120
                )
                if timed_out:
                    print(f"    WARNING: Dedup fitness timed out (120s) — skipping")
                elif dedup_val is not None:
                    result["dedup"] = round(dedup_val.get("score", 0.0), 4)
    except Exception as e:
        print(f"    WARNING: Dedup fitness error: {e}")

    return result


def compute_powl_analysis(df_mapped: pd.DataFrame) -> dict:
    """Run POWL shadow workflow analysis."""
    result = {
        "shadow_nodes": 0, "full_nodes": 0, "shadow_activities": 0,
        "cluster_count": 0,
        "cluster_types": {"recurring": 0, "concurrent": 0, "isolated": 0, "sequential": 0},
        "avg_cluster_duration_s": 0.0,
    }

    try:
        # Shadow POWL
        shadow_powl, shadow_count = discover_shadow_powl(df_mapped)
        if shadow_powl:
            stats = get_powl_statistics(shadow_powl)
            result["shadow_nodes"] = stats.get("node_count", 0)
            unique_acts = stats.get("unique_activities", set())
            result["shadow_activities"] = len(unique_acts) if isinstance(unique_acts, set) else unique_acts

        # Full POWL
        full_powl, full_count = discover_full_powl(df_mapped)
        if full_powl:
            full_stats = get_powl_statistics(full_powl)
            result["full_nodes"] = full_stats.get("node_count", 0)

        # Shadow pattern clusters
        clusters = detect_shadow_clusters(df_mapped)
        classified = classify_shadow_patterns(clusters)
        summary = get_pattern_summary(classified)

        result["cluster_count"] = summary.get("total_clusters", 0)
        pattern_counts = summary.get("pattern_counts", {})
        result["cluster_types"] = {
            "recurring": pattern_counts.get("recurring", 0),
            "concurrent": pattern_counts.get("concurrent", 0),
            "isolated": pattern_counts.get("isolated", 0),
            "sequential": pattern_counts.get("sequential", 0),
        }
        if classified:
            durations = [c.get("duration", 0) for c in classified]
            result["avg_cluster_duration_s"] = round(sum(durations) / len(durations), 1)

    except Exception as e:
        print(f"    WARNING: POWL analysis error: {e}")

    return result


def compute_declare_analysis(
    df_mapped: pd.DataFrame,
    activities: list[str],
) -> dict:
    """Run Declare conformance against Robert's Rules."""
    result = {
        "constraints_checked": 0, "satisfied": 0, "violated": 0,
        "compliance_score": 0.0, "grade": "F", "violations": [],
    }

    try:
        conformance_results = check_roberts_rules_conformance(df_mapped, activities)
        if not conformance_results:
            return result

        score_result = compute_procedural_compliance_score(conformance_results)
        violations = classify_violations(conformance_results)

        result["constraints_checked"] = score_result.get("total_checked", 0)
        result["satisfied"] = score_result.get("satisfied_count", 0)
        result["violated"] = score_result.get("violated_count", 0)
        result["compliance_score"] = round(score_result.get("score", 0.0), 1)
        result["grade"] = score_result.get("grade", "F")

        # Top violations for conformance.json
        result["violations"] = [
            {
                "template": v.get("template", ""),
                "a": v.get("activity_a", ""),
                "b": v.get("activity_b", ""),
                "severity": v.get("severity", "low"),
            }
            for v in violations[:5]  # Top 5
        ]

    except Exception as e:
        print(f"    WARNING: Declare analysis error: {e}")

    return result


def compute_deviance_analysis(
    df_mapped: pd.DataFrame,
    declare_violations: list[dict] | None,
    api_key: str | None,
) -> dict:
    """Run deviance classification on shadow events."""
    result = {
        "benign": 0, "violation": 0, "efficiency": 0,
        "innovation": 0, "disruption": 0, "unknown": 0,
        "severity_pct": 0.0,
    }

    try:
        classifier = DevianceClassifier(api_key=api_key)
        classified = classifier.classify(df_mapped, declare_violations)
        summary = classifier.generate_deviance_summary(classified)

        cat_dist = summary.get("category_distribution", {})
        result["benign"] = cat_dist.get("benign", 0)
        result["violation"] = cat_dist.get("violation", 0)
        result["efficiency"] = cat_dist.get("efficient", 0)
        result["innovation"] = cat_dist.get("innovation", 0)
        result["disruption"] = cat_dist.get("disruption", 0)
        result["unknown"] = cat_dist.get("unknown", 0)
        result["severity_pct"] = round(summary.get("severity_score", 0.0), 2)

        # Save classified events
        classified.to_csv(
            os.path.join(os.path.dirname(df_mapped.attrs.get("_path", "")), "classified_events.csv"),
            index=False,
        ) if hasattr(df_mapped, "attrs") else None

    except Exception as e:
        print(f"    WARNING: Deviance analysis error: {e}")

    return result


def run_layer4(
    meeting: dict,
    df_mapped: pd.DataFrame,
    activities: list[str],
    bpmn_obj,
    api_key: str | None,
    engine: ComplianceEngine,
    extraction_info: dict,
    params: dict,
    layer3_result: dict,
) -> dict:
    """Full conformance analysis -> conformance.json."""
    t0 = time.time()

    print_subsection("BPMN Fitness")
    fitness = compute_bpmn_fitness(df_mapped, bpmn_obj, activities, engine)
    print(f"    Fitness: raw={fitness['raw']:.4f}, dedup={fitness['dedup']:.4f}")

    print_subsection("POWL Analysis")
    powl = compute_powl_analysis(df_mapped)
    print(f"    POWL: {powl['cluster_count']} clusters, {powl['shadow_nodes']} shadow nodes")

    print_subsection("Declare Conformance")
    declare = compute_declare_analysis(df_mapped, activities)
    print(f"    Declare: {declare['compliance_score']}/100 (Grade {declare['grade']})")

    print_subsection("Deviance Classification")
    declare_violations = declare.get("violations", [])
    deviance = compute_deviance_analysis(df_mapped, declare_violations, api_key)
    print(f"    Deviance: {deviance['benign']} benign, {deviance['violation']} violations")

    # Build conformance.json
    formal_count = layer3_result.get("formal_count", 0)
    shadow_count = layer3_result.get("shadow_count", 0)
    total_events = formal_count + shadow_count

    conformance = {
        "meeting_id": meeting["folder_name"],
        "city": meeting["city"],
        "date": meeting["date"],
        "duration_seconds": extraction_info.get("duration_seconds", 0),
        "agenda_items": len(activities),

        "extraction": {
            "raw_event_count": extraction_info.get("raw_event_count", 0),
            "sources": extraction_info.get("source_distribution", {}),
            "whisper_model": params.get("whisper_model", "small"),
            "processing_time_seconds": round(extraction_info.get("elapsed_seconds", 0), 1),
        },

        "variant": f"llm_{params['openai_model'].replace('-', '_')}",
        "params": {
            "window_seconds": params["window_seconds"],
            "overlap_ratio": params["overlap_ratio"],
            "sbert_threshold": params["sbert_threshold"],
            "sbert_model": params["sbert_model"],
        },

        "fitness": fitness,
        "match_rate": fitness["match_rate"],
        "agenda_coverage": fitness["agenda_coverage"],
        "agenda_coverage_pct": fitness["agenda_coverage_pct"],

        "event_allocation": {
            "formal": formal_count,
            "shadow": shadow_count,
            "shadow_pct": round(shadow_count / total_events, 3) if total_events > 0 else 0.0,
        },

        "powl": powl,
        "declare": declare,
        "deviance": deviance,

        "processed_at": datetime.now().isoformat(),
    }

    # Save conformance.json
    conf_path = os.path.join(meeting["path"], "conformance.json")
    with open(conf_path, "w", encoding="utf-8") as f:
        json.dump(conformance, f, indent=2)

    elapsed = time.time() - t0
    print(f"    Layer 4: completed in {elapsed:.0f}s")
    return conformance


# ============================================================
# Single meeting orchestrator
# ============================================================

def process_single_meeting(
    meeting: dict,
    args: argparse.Namespace,
    engine: ComplianceEngine,
) -> dict | None:
    """Process one meeting through all layers."""
    print_section(f"{meeting['folder_name']}")

    if not meeting["has_agenda"]:
        print("  SKIP: No agenda.txt found")
        return None

    status = check_layer_status(meeting["path"])

    # Early return if all layers are already complete
    if status["layer4_done"] and not getattr(args, 'force_layer4', False):
        print("  All layers complete — skipping")
        conf_path = os.path.join(meeting["path"], "conformance.json")
        with open(conf_path) as f:
            return json.load(f)

    params = {
        "window_seconds": args.window_seconds,
        "overlap_ratio": args.overlap_ratio,
        "min_events": args.min_events,
        "min_label_support": args.min_label_support,
        "sbert_threshold": args.sbert_threshold,
        "sbert_model": args.sbert_model,
        "openai_model": args.openai_model,
        "shadow_min_ratio": 0.15,
        "openai_timeout": 60,
        "max_windows": 150,
        "whisper_model": args.whisper_model,
    }

    try:
        # --- Layer 2: Video Processing ---
        extraction_info = {}
        df_raw = None

        if status["layer2_done"]:
            raw_csv = os.path.join(meeting["path"], "raw_events.csv")
            df_raw = pd.read_csv(raw_csv)
            print(f"  Layer 2: loaded {len(df_raw)} cached events")

            # Load source distribution if available
            sd_path = os.path.join(meeting["path"], "source_distribution.json")
            source_dist = {}
            if os.path.exists(sd_path):
                with open(sd_path) as f:
                    source_dist = json.load(f)
            else:
                source_dist = compute_source_distribution(df_raw)

            # Estimate duration
            duration_s = 0
            if "timestamp" in df_raw.columns and len(df_raw) > 0:
                try:
                    duration_s = int(ts_to_seconds(str(df_raw["timestamp"].iloc[-1])))
                except Exception:
                    pass

            extraction_info = {
                "raw_event_count": len(df_raw),
                "source_distribution": source_dist,
                "duration_seconds": duration_s,
                "elapsed_seconds": 0,
            }

        elif meeting["has_video"]:
            extraction_info = run_layer2(
                meeting, args.api_key,
                use_local_whisper=args.local_whisper,
                whisper_model=args.whisper_model,
            )
            df_raw = extraction_info["df_raw"]

        else:
            print("  SKIP: No video.mp4 found")
            return None

        if df_raw is None or df_raw.empty:
            print("  SKIP: No raw events")
            return None

        # --- Layer 3: LLM Abstraction + SBERT Mapping ---
        if status["layer3_done"] and not args.force_layer3:
            mapped_csv = os.path.join(meeting["path"], "variant_llm", "mapped_events.csv")
            df_mapped = pd.read_csv(mapped_csv)
            print(f"  Layer 3: loaded {len(df_mapped)} cached mapped events")

            # We need activities and bpmn_obj for Layer 4
            agenda_path = os.path.join(meeting["path"], "agenda.txt")
            with open(agenda_path, "r", encoding="utf-8") as f:
                agenda_text = f.read()
            _, activities, bpmn_obj = generate_agenda_bpmn(agenda_text, args.api_key)

            act_col = "mapped_activity" if "mapped_activity" in df_mapped.columns else "activity_name"
            shadow_mask = df_mapped[act_col].str.startswith("Deviation:", na=False)
            layer3_result = {
                "activities": activities,
                "bpmn_obj": bpmn_obj,
                "df_mapped": df_mapped,
                "formal_count": int((~shadow_mask).sum()),
                "shadow_count": int(shadow_mask.sum()),
            }
        else:
            agenda_path = os.path.join(meeting["path"], "agenda.txt")
            with open(agenda_path, "r", encoding="utf-8") as f:
                agenda_text = f.read()

            layer3_result = run_layer3(
                meeting, df_raw, agenda_text, args.api_key, params, engine,
            )
            df_mapped = layer3_result["df_mapped"]
            activities = layer3_result["activities"]
            bpmn_obj = layer3_result["bpmn_obj"]

        # --- Layer 4: Conformance Analysis ---
        if status["layer4_done"] and not args.force_layer4:
            print("  Layer 4: already complete (use --force-layer4 to re-run)")
            conf_path = os.path.join(meeting["path"], "conformance.json")
            with open(conf_path) as f:
                return json.load(f)

        conformance = run_layer4(
            meeting, df_mapped, activities, bpmn_obj,
            args.api_key, engine, extraction_info, params, layer3_result,
        )
        return conformance

    except Exception as e:
        error_msg = f"ERROR processing {meeting['folder_name']}: {e}"
        print(f"  {error_msg}")
        # Save error info
        error_info = {
            "meeting": meeting["folder_name"],
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
        }
        error_path = os.path.join(meeting["path"], "processing_error.json")
        with open(error_path, "w") as f:
            json.dump(error_info, f, indent=2)
        return None


# ============================================================
# Cross-meeting aggregation
# ============================================================

def aggregate_results(meetings_dir: str) -> dict:
    """Load all conformance.json files and build cross-meeting summary."""
    agg_dir = os.path.join(meetings_dir, "_aggregated")
    os.makedirs(agg_dir, exist_ok=True)

    all_conformance = []
    for folder_name in sorted(os.listdir(meetings_dir)):
        conf_path = os.path.join(meetings_dir, folder_name, "conformance.json")
        if os.path.exists(conf_path):
            try:
                with open(conf_path) as f:
                    all_conformance.append(json.load(f))
            except json.JSONDecodeError:
                print(f"  WARNING: Skipping corrupted {conf_path}")
                continue

    if not all_conformance:
        print("  No conformance.json files found for aggregation")
        return {}

    # Build metrics CSV
    rows = []
    for c in all_conformance:
        fitness = c.get("fitness", {})
        declare = c.get("declare", {})
        deviance = c.get("deviance", {})
        event_alloc = c.get("event_allocation", {})
        extraction = c.get("extraction", {})
        powl = c.get("powl", {})

        rows.append({
            "meeting_id": c.get("meeting_id", ""),
            "city": c.get("city", ""),
            "date": c.get("date", ""),
            "agenda_items": c.get("agenda_items", 0),
            "raw_events": extraction.get("raw_event_count", 0),
            "duration_seconds": c.get("duration_seconds", 0),
            "fitness_raw": fitness.get("raw", 0),
            "fitness_dedup": fitness.get("dedup", 0),
            "match_rate": c.get("match_rate", 0),
            "agenda_coverage": c.get("agenda_coverage", 0),
            "agenda_coverage_pct": c.get("agenda_coverage_pct", 0),
            "formal_events": event_alloc.get("formal", 0),
            "shadow_events": event_alloc.get("shadow", 0),
            "shadow_pct": event_alloc.get("shadow_pct", 0),
            "declare_score": declare.get("compliance_score", 0),
            "declare_grade": declare.get("grade", ""),
            "declare_satisfied": declare.get("satisfied", 0),
            "declare_violated": declare.get("violated", 0),
            "deviance_benign": deviance.get("benign", 0),
            "deviance_violation": deviance.get("violation", 0),
            "deviance_severity_pct": deviance.get("severity_pct", 0),
            "powl_clusters": powl.get("cluster_count", 0),
            "powl_shadow_nodes": powl.get("shadow_nodes", 0),
        })

    df_metrics = pd.DataFrame(rows)
    csv_path = os.path.join(agg_dir, "metrics.csv")
    df_metrics.to_csv(csv_path, index=False)

    # Compute summary stats
    cities = df_metrics["city"].unique().tolist()
    city_summaries = {}
    for city in cities:
        city_df = df_metrics[df_metrics["city"] == city]
        city_summaries[city] = {
            "count": len(city_df),
            "avg_fitness_dedup": round(city_df["fitness_dedup"].mean(), 4),
            "avg_shadow_pct": round(city_df["shadow_pct"].mean(), 4),
            "avg_declare_score": round(city_df["declare_score"].mean(), 1),
            "avg_agenda_coverage_pct": round(city_df["agenda_coverage_pct"].mean(), 4),
        }

    summary = {
        "total_meetings": len(all_conformance),
        "cities": cities,
        "meetings_by_city": {c: s["count"] for c, s in city_summaries.items()},
        "avg_fitness_raw": round(df_metrics["fitness_raw"].mean(), 4),
        "avg_fitness_dedup": round(df_metrics["fitness_dedup"].mean(), 4),
        "avg_shadow_pct": round(df_metrics["shadow_pct"].mean(), 4),
        "avg_declare_score": round(df_metrics["declare_score"].mean(), 1),
        "city_summaries": city_summaries,
        "aggregated_at": datetime.now().isoformat(),
    }

    summary_path = os.path.join(agg_dir, "cross_meeting_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Aggregated {len(all_conformance)} meetings -> {csv_path}")
    return summary


# ============================================================
# CLI and main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch analysis of downloaded city council meetings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--meetings-dir", default="meetings",
                        help="Path to meetings/ folder")
    parser.add_argument("--api-key", required=True,
                        help="OpenAI API key")

    # Video processing
    parser.add_argument("--local-whisper", action="store_true", default=True,
                        help="Use local Whisper model")
    parser.add_argument("--whisper-model", default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Local Whisper model size")
    parser.add_argument("--skip-video", action="store_true",
                        help="Skip Layer 2 if raw_events.csv exists")

    # LLM abstraction parameters
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--min-events", type=int, default=2)
    parser.add_argument("--min-label-support", type=int, default=1)
    parser.add_argument("--sbert-threshold", type=float, default=0.35)
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--openai-model", default="gpt-4o-mini")

    # Filtering
    parser.add_argument("--only-cities", default=None,
                        help="Comma-separated city filter (e.g. seattle,denver)")
    parser.add_argument("--only-meeting", default=None,
                        help="Process a single meeting folder name")
    parser.add_argument("--max-meetings", type=int, default=None,
                        help="Limit to N meetings")

    # Control
    parser.add_argument("--force-layer3", action="store_true",
                        help="Re-run Layer 3 even if results exist")
    parser.add_argument("--force-layer4", action="store_true",
                        help="Re-run Layer 4 even if results exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs only, do not process")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    print_section("Meeting Process Twin — Batch Analysis")

    # Discover meetings
    only_cities = args.only_cities.split(",") if args.only_cities else None
    meetings = discover_meetings(args.meetings_dir, only_cities, args.only_meeting)

    if args.max_meetings:
        meetings = meetings[:args.max_meetings]

    print(f"  Found {len(meetings)} meetings")
    if not meetings:
        print("  No meetings to process")
        return

    # Count by city
    city_counts = Counter(m["city"] for m in meetings)
    for city, count in sorted(city_counts.items()):
        print(f"    {city}: {count}")

    # Report layer status
    print(f"\n  Layer status:")
    statuses = [check_layer_status(m["path"]) for m in meetings]
    l2_done = sum(1 for s in statuses if s["layer2_done"])
    l3_done = sum(1 for s in statuses if s["layer3_done"])
    l4_done = sum(1 for s in statuses if s["layer4_done"])
    has_video = sum(1 for m in meetings if m["has_video"])
    has_agenda = sum(1 for m in meetings if m["has_agenda"])
    print(f"    Has video:     {has_video}/{len(meetings)}")
    print(f"    Has agenda:    {has_agenda}/{len(meetings)}")
    print(f"    Layer 2 done:  {l2_done}/{len(meetings)}")
    print(f"    Layer 3 done:  {l3_done}/{len(meetings)}")
    print(f"    Layer 4 done:  {l4_done}/{len(meetings)}")

    need_l2 = has_video - l2_done if not args.skip_video else 0
    need_l3 = len(meetings) - l3_done
    print(f"\n  To process:")
    print(f"    Layer 2 (video): {need_l2} meetings (~{need_l2 * 15} min)")
    print(f"    Layer 3 (LLM):   {need_l3} meetings (~{need_l3 * 3} min)")
    print(f"    Layer 4 (conformance): {len(meetings) - l4_done} meetings (~{(len(meetings) - l4_done)} min)")

    if args.dry_run:
        print("\n  (Dry run — nothing processed)")
        return

    # Initialize shared engine
    print(f"\n  Loading SBERT model: {args.sbert_model}...")
    engine = ComplianceEngine(sbert_model_name=args.sbert_model)

    # Process each meeting
    results = []
    success = 0
    failed = 0
    skipped = 0

    for i, meeting in enumerate(meetings, 1):
        print(f"\n  [{i}/{len(meetings)}]", end="")
        t0 = time.time()

        conformance = process_single_meeting(meeting, args, engine)

        elapsed = time.time() - t0

        if conformance:
            results.append(conformance)
            success += 1
            print(f"  Done ({elapsed:.0f}s)")
        else:
            if not meeting["has_video"] or not meeting["has_agenda"]:
                skipped += 1
            else:
                failed += 1

        # Save incremental progress
        progress = {
            "processed": i,
            "total": len(meetings),
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "updated_at": datetime.now().isoformat(),
        }
        agg_dir = os.path.join(args.meetings_dir, "_aggregated")
        os.makedirs(agg_dir, exist_ok=True)
        with open(os.path.join(agg_dir, "batch_progress.json"), "w") as f:
            json.dump(progress, f, indent=2)

    # Aggregate
    print_section("Aggregation")
    summary = aggregate_results(args.meetings_dir)

    # Final summary
    print_section("Batch Complete")
    print(f"  Processed: {success} success, {failed} failed, {skipped} skipped")
    if summary:
        print(f"  Avg fitness (dedup): {summary.get('avg_fitness_dedup', 0):.4f}")
        print(f"  Avg Declare score:   {summary.get('avg_declare_score', 0):.1f}")
        print(f"  Avg shadow %:        {summary.get('avg_shadow_pct', 0):.1%}")
    print(f"\n  Results: {os.path.abspath(args.meetings_dir)}/_aggregated/")
    print(f"  Next: python generate_figures.py --meetings-dir {args.meetings_dir}")


if __name__ == "__main__":
    main()
