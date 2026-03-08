"""Evaluation utilities comparing keyword-based vs LLM-based structured extraction.

Provides side-by-side metrics for understanding the quality tradeoffs between
fast keyword extraction (from keyword_rules.py) and richer LLM-based structured
extraction (from llm_extractor.py).

Example usage:
    from research.structured_extraction.extraction_eval import (
        compare_extractions,
        compute_coverage_report,
        evaluate_actor_extraction,
    )

    report = compare_extractions(
        keyword_events=keyword_df,
        structured_events=structured_event_list,
        agenda_activities=["Call to Order", "Roll Call", ...],
    )
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from typing import Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project-level time utilities
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.time_utils import ts_to_seconds, seconds_to_ts  # noqa: E402

# ---------------------------------------------------------------------------
# Schema imports
# ---------------------------------------------------------------------------
from research.structured_extraction.schema import (  # noqa: E402
    MeetingEvent,
    ExtractionBatch,
)

# ---------------------------------------------------------------------------
# Optional SBERT for semantic similarity
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

logger = logging.getLogger(__name__)


def compare_extractions(
    keyword_events: pd.DataFrame,
    structured_events: Union[list[MeetingEvent], ExtractionBatch],
    agenda_activities: list[str],
    sbert_model_name: str = "all-MiniLM-L6-v2",
) -> dict:
    """Compare keyword extraction vs LLM structured extraction.

    Provides a comprehensive comparison of both extraction methods across
    multiple dimensions: volume, diversity, coverage, temporal alignment,
    and actor attribution.

    Args:
        keyword_events:    DataFrame from the standard keyword extraction pipeline
                           (columns: timestamp, activity_name, source, details,
                           original_text, mapped_activity).
        structured_events: Either a list of MeetingEvent objects or an
                           ExtractionBatch from StructuredEventExtractor.
        agenda_activities: List of formal agenda item labels.
        sbert_model_name:  SBERT model for semantic similarity comparisons.

    Returns:
        Dictionary containing::

            {
                "keyword": {
                    "total_events": int,
                    "unique_labels": int,
                    "label_distribution": dict,
                    "agenda_coverage": float,
                    "covered_items": list[str],
                    "uncovered_items": list[str],
                    "temporal_span": dict,
                },
                "structured": {
                    "total_events": int,
                    "unique_labels": int,
                    "unique_actors": int,
                    "actor_list": list[str],
                    "event_type_distribution": dict,
                    "avg_confidence": float,
                    "agenda_coverage": float,
                    "covered_items": list[str],
                    "uncovered_items": list[str],
                    "temporal_span": dict,
                },
                "comparison": {
                    "event_ratio": float,          # structured / keyword event count
                    "label_diversity_ratio": float, # structured / keyword unique labels
                    "coverage_delta": float,        # structured - keyword coverage
                    "actor_enrichment": int,        # unique actors from structured (not in keyword)
                    "semantic_overlap": float,      # average pairwise similarity of labels
                    "recommendation": str,
                },
            }
    """
    # Normalize structured_events to list of MeetingEvent
    if isinstance(structured_events, ExtractionBatch):
        event_list = structured_events.all_events
    else:
        event_list = structured_events

    # --- Keyword extraction analysis ---
    kw_analysis = _analyze_keyword_events(keyword_events, agenda_activities)

    # --- Structured extraction analysis ---
    st_analysis = _analyze_structured_events(event_list, agenda_activities)

    # --- Cross-comparison ---
    comparison = _compute_comparison(kw_analysis, st_analysis, sbert_model_name)

    return {
        "keyword": kw_analysis,
        "structured": st_analysis,
        "comparison": comparison,
    }


def compute_coverage_report(
    events: Union[pd.DataFrame, list[MeetingEvent]],
    agenda_activities: list[str],
    sbert_model_name: str = "all-MiniLM-L6-v2",
    sbert_threshold: float = 0.45,
) -> dict:
    """Compute detailed agenda coverage for a set of events.

    Uses both exact string matching and SBERT semantic similarity to
    determine which agenda items are covered by the extracted events.

    Args:
        events:           Either a DataFrame or list of MeetingEvent objects.
        agenda_activities: List of formal agenda item labels.
        sbert_model_name:  SBERT model name for similarity matching.
        sbert_threshold:   Minimum cosine similarity for a match.

    Returns:
        Dictionary with per-item coverage details::

            {
                "total_agenda_items": int,
                "covered_count": int,
                "coverage_pct": float,
                "per_item": {
                    "Call to Order": {
                        "covered": bool,
                        "match_type": "exact" | "semantic" | "none",
                        "best_match_score": float,
                        "matching_events": int,
                    },
                    ...
                },
            }
    """
    # Normalize to list of label strings
    if isinstance(events, pd.DataFrame):
        labels = events["activity_name"].dropna().tolist()
        if "mapped_activity" in events.columns:
            mapped = events["mapped_activity"].dropna().tolist()
            labels.extend([m for m in mapped if m])
    else:
        labels = [e.action for e in events]
        labels.extend([
            f"{e.actor} {e.action} {e.object or ''}" for e in events
        ])

    # Remove duplicates for efficiency
    unique_labels = list(set(labels))

    # Load SBERT if available
    sbert_model = None
    if _SBERT_AVAILABLE and unique_labels:
        try:
            sbert_model = SentenceTransformer(sbert_model_name)
        except Exception as e:
            logger.warning("Could not load SBERT model: %s", e)

    per_item: dict[str, dict] = {}
    covered_count = 0

    for agenda_item in agenda_activities:
        item_lower = agenda_item.lower().strip()

        # Exact match check
        exact_matches = [
            l for l in labels if item_lower in l.lower() or l.lower() in item_lower
        ]

        if exact_matches:
            per_item[agenda_item] = {
                "covered": True,
                "match_type": "exact",
                "best_match_score": 1.0,
                "matching_events": len(exact_matches),
            }
            covered_count += 1
            continue

        # SBERT semantic match
        if sbert_model is not None and unique_labels:
            agenda_emb = sbert_model.encode(agenda_item, convert_to_tensor=True)
            label_embs = sbert_model.encode(unique_labels, convert_to_tensor=True)
            scores = sbert_util.cos_sim(agenda_emb, label_embs)[0]
            best_score = float(scores.max())

            if best_score >= sbert_threshold:
                matching_count = int((scores >= sbert_threshold).sum())
                per_item[agenda_item] = {
                    "covered": True,
                    "match_type": "semantic",
                    "best_match_score": round(best_score, 4),
                    "matching_events": matching_count,
                }
                covered_count += 1
                continue

        # No match
        per_item[agenda_item] = {
            "covered": False,
            "match_type": "none",
            "best_match_score": 0.0,
            "matching_events": 0,
        }

    total = len(agenda_activities)
    return {
        "total_agenda_items": total,
        "covered_count": covered_count,
        "coverage_pct": round(covered_count / total * 100, 1) if total > 0 else 0.0,
        "per_item": per_item,
    }


def evaluate_actor_extraction(structured_events: list[MeetingEvent]) -> dict:
    """Evaluate the quality of actor (speaker) extraction from structured events.

    Args:
        structured_events: List of MeetingEvent objects.

    Returns:
        Dictionary containing::

            {
                "total_events": int,
                "events_with_actor": int,
                "actor_extraction_rate": float,
                "unique_actors": list[str],
                "actor_event_counts": dict[str, int],
                "role_patterns": dict[str, int],   # e.g. {"Mayor": 5, "Council Member": 12}
                "anonymous_events": int,
            }
    """
    if not structured_events:
        return {
            "total_events": 0,
            "events_with_actor": 0,
            "actor_extraction_rate": 0.0,
            "unique_actors": [],
            "actor_event_counts": {},
            "role_patterns": {},
            "anonymous_events": 0,
        }

    total = len(structured_events)
    with_actor = [e for e in structured_events if e.actor]
    without_actor = total - len(with_actor)

    # Count actors
    actor_counts = Counter(e.actor for e in with_actor)
    unique_actors = sorted(actor_counts.keys())

    # Extract role patterns (first word or "Council Member" prefix)
    role_counts: Counter = Counter()
    for actor in actor_counts:
        # Detect common role patterns
        actor_lower = actor.lower()
        if "mayor" in actor_lower:
            role_counts["Mayor"] += actor_counts[actor]
        elif "council member" in actor_lower or "councilmember" in actor_lower:
            role_counts["Council Member"] += actor_counts[actor]
        elif "clerk" in actor_lower:
            role_counts["City Clerk"] += actor_counts[actor]
        elif "attorney" in actor_lower:
            role_counts["City Attorney"] += actor_counts[actor]
        elif "manager" in actor_lower:
            role_counts["City Manager"] += actor_counts[actor]
        elif "public" in actor_lower or "citizen" in actor_lower:
            role_counts["Public/Citizen"] += actor_counts[actor]
        else:
            role_counts["Other"] += actor_counts[actor]

    return {
        "total_events": total,
        "events_with_actor": len(with_actor),
        "actor_extraction_rate": round(len(with_actor) / total, 4),
        "unique_actors": unique_actors,
        "actor_event_counts": dict(actor_counts),
        "role_patterns": dict(role_counts),
        "anonymous_events": without_actor,
    }


def compute_temporal_alignment(
    keyword_events: pd.DataFrame,
    structured_events: list[MeetingEvent],
    bucket_seconds: int = 60,
) -> pd.DataFrame:
    """Compare the temporal distribution of keyword vs structured events.

    Groups events into time buckets and compares counts, useful for
    understanding if one method captures events in certain meeting phases
    better than the other.

    Args:
        keyword_events:    Standard pipeline DataFrame.
        structured_events: List of MeetingEvent objects.
        bucket_seconds:    Width of each time bucket in seconds.

    Returns:
        DataFrame with columns: bucket_start, bucket_end, keyword_count,
        structured_count, delta.
    """
    def _bucket_events(timestamps: list[int], max_sec: int) -> Counter:
        buckets: Counter = Counter()
        for ts in timestamps:
            bucket = (ts // bucket_seconds) * bucket_seconds
            buckets[bucket] += 1
        return buckets

    # Extract timestamps in seconds
    kw_timestamps = keyword_events["timestamp"].apply(ts_to_seconds).tolist()
    st_timestamps = [ts_to_seconds(e.timestamp) for e in structured_events]

    all_ts = kw_timestamps + st_timestamps
    max_sec = max(all_ts) if all_ts else 0

    kw_buckets = _bucket_events(kw_timestamps, max_sec)
    st_buckets = _bucket_events(st_timestamps, max_sec)

    # Build all bucket starts
    all_bucket_starts = sorted(set(kw_buckets.keys()) | set(st_buckets.keys()))

    rows = []
    for bs in all_bucket_starts:
        kw_count = kw_buckets.get(bs, 0)
        st_count = st_buckets.get(bs, 0)
        rows.append({
            "bucket_start": seconds_to_ts(bs),
            "bucket_end": seconds_to_ts(bs + bucket_seconds),
            "keyword_count": kw_count,
            "structured_count": st_count,
            "delta": st_count - kw_count,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal analysis helpers
# ---------------------------------------------------------------------------

def _analyze_keyword_events(
    keyword_events: pd.DataFrame,
    agenda_activities: list[str],
) -> dict:
    """Analyze keyword-extracted event DataFrame."""
    if keyword_events.empty:
        return {
            "total_events": 0,
            "unique_labels": 0,
            "label_distribution": {},
            "agenda_coverage": 0.0,
            "covered_items": [],
            "uncovered_items": list(agenda_activities),
            "temporal_span": {"start": "00:00:00", "end": "00:00:00", "duration_seconds": 0},
        }

    labels = keyword_events["activity_name"].dropna().tolist()
    label_dist = Counter(labels)

    # Coverage: how many agenda items have at least one matching event?
    covered = set()
    mapped_col = "mapped_activity" if "mapped_activity" in keyword_events.columns else None

    for agenda_item in agenda_activities:
        item_lower = agenda_item.lower()
        # Check activity_name
        if any(item_lower in l.lower() or l.lower() in item_lower for l in labels):
            covered.add(agenda_item)
            continue
        # Check mapped_activity
        if mapped_col:
            mapped = keyword_events[mapped_col].dropna().tolist()
            if any(item_lower in m.lower() or m.lower() in item_lower for m in mapped):
                covered.add(agenda_item)

    uncovered = [a for a in agenda_activities if a not in covered]

    # Temporal span
    ts_seconds = keyword_events["timestamp"].apply(ts_to_seconds)
    start_sec = int(ts_seconds.min()) if not ts_seconds.empty else 0
    end_sec = int(ts_seconds.max()) if not ts_seconds.empty else 0

    return {
        "total_events": len(keyword_events),
        "unique_labels": len(label_dist),
        "label_distribution": dict(label_dist.most_common(30)),
        "agenda_coverage": round(len(covered) / len(agenda_activities) * 100, 1) if agenda_activities else 0.0,
        "covered_items": sorted(covered),
        "uncovered_items": uncovered,
        "temporal_span": {
            "start": seconds_to_ts(start_sec),
            "end": seconds_to_ts(end_sec),
            "duration_seconds": end_sec - start_sec,
        },
    }


def _analyze_structured_events(
    structured_events: list[MeetingEvent],
    agenda_activities: list[str],
) -> dict:
    """Analyze LLM-extracted structured events."""
    if not structured_events:
        return {
            "total_events": 0,
            "unique_labels": 0,
            "unique_actors": 0,
            "actor_list": [],
            "event_type_distribution": {},
            "avg_confidence": 0.0,
            "agenda_coverage": 0.0,
            "covered_items": [],
            "uncovered_items": list(agenda_activities),
            "temporal_span": {"start": "00:00:00", "end": "00:00:00", "duration_seconds": 0},
        }

    # Build labels from (actor, action, object)
    labels = [e.action for e in structured_events]
    label_dist = Counter(labels)

    # Actors
    actors = sorted({e.actor for e in structured_events if e.actor})

    # Event types
    type_dist = Counter(e.event_type for e in structured_events)

    # Confidence
    avg_conf = sum(e.confidence for e in structured_events) / len(structured_events)

    # Coverage: check if any event's action/object matches agenda items
    covered = set()
    for agenda_item in agenda_activities:
        item_lower = agenda_item.lower()
        for event in structured_events:
            full_text = f"{event.actor or ''} {event.action} {event.object or ''}".lower()
            if item_lower in full_text or any(
                word in full_text for word in item_lower.split() if len(word) > 3
            ):
                covered.add(agenda_item)
                break

    uncovered = [a for a in agenda_activities if a not in covered]

    # Temporal span
    ts_seconds = [ts_to_seconds(e.timestamp) for e in structured_events]
    start_sec = min(ts_seconds) if ts_seconds else 0
    end_sec = max(ts_seconds) if ts_seconds else 0

    return {
        "total_events": len(structured_events),
        "unique_labels": len(label_dist),
        "unique_actors": len(actors),
        "actor_list": actors,
        "event_type_distribution": dict(type_dist),
        "avg_confidence": round(avg_conf, 4),
        "agenda_coverage": round(len(covered) / len(agenda_activities) * 100, 1) if agenda_activities else 0.0,
        "covered_items": sorted(covered),
        "uncovered_items": uncovered,
        "temporal_span": {
            "start": seconds_to_ts(start_sec),
            "end": seconds_to_ts(end_sec),
            "duration_seconds": end_sec - start_sec,
        },
    }


def _compute_comparison(
    kw_analysis: dict,
    st_analysis: dict,
    sbert_model_name: str,
) -> dict:
    """Compute cross-method comparison metrics."""
    kw_total = kw_analysis.get("total_events", 0)
    st_total = st_analysis.get("total_events", 0)
    kw_labels = kw_analysis.get("unique_labels", 0)
    st_labels = st_analysis.get("unique_labels", 0)
    kw_cov = kw_analysis.get("agenda_coverage", 0.0)
    st_cov = st_analysis.get("agenda_coverage", 0.0)

    event_ratio = st_total / kw_total if kw_total > 0 else float("inf") if st_total > 0 else 0.0
    label_ratio = st_labels / kw_labels if kw_labels > 0 else float("inf") if st_labels > 0 else 0.0
    coverage_delta = st_cov - kw_cov

    # Actor enrichment: structured extraction provides actor info that keywords do not
    actor_enrichment = st_analysis.get("unique_actors", 0)

    # Semantic overlap between label sets
    semantic_overlap = 0.0
    kw_label_list = list(kw_analysis.get("label_distribution", {}).keys())
    st_label_list = list(st_analysis.get("event_type_distribution", {}).keys())

    if _SBERT_AVAILABLE and kw_label_list and st_label_list:
        try:
            model = SentenceTransformer(sbert_model_name)
            kw_embs = model.encode(kw_label_list[:50], convert_to_tensor=True)
            st_labels_for_sim = [
                e.action for e in []  # placeholder; we use label_dist keys instead
            ] or list(kw_analysis.get("label_distribution", {}).keys())[:50]

            # Use the actual action labels from structured analysis
            st_action_labels = list(st_analysis.get("label_distribution", {}).keys()) if "label_distribution" in st_analysis else st_label_list
            if st_action_labels:
                st_embs = model.encode(st_action_labels[:50], convert_to_tensor=True)
                sim_matrix = sbert_util.cos_sim(kw_embs, st_embs)
                # Average of max similarity per keyword label
                if sim_matrix.numel() > 0:
                    max_sims = sim_matrix.max(dim=1).values
                    semantic_overlap = float(max_sims.mean())
        except Exception as e:
            logger.warning("SBERT comparison failed: %s", e)

    # Generate recommendation
    recommendation = _generate_recommendation(
        kw_total, st_total, kw_cov, st_cov, actor_enrichment, coverage_delta,
    )

    return {
        "event_ratio": round(event_ratio, 2),
        "label_diversity_ratio": round(label_ratio, 2),
        "coverage_delta": round(coverage_delta, 1),
        "actor_enrichment": actor_enrichment,
        "semantic_overlap": round(semantic_overlap, 4),
        "recommendation": recommendation,
    }


def _generate_recommendation(
    kw_total: int,
    st_total: int,
    kw_cov: float,
    st_cov: float,
    actor_count: int,
    coverage_delta: float,
) -> str:
    """Generate a human-readable recommendation based on comparison."""
    parts = []

    if coverage_delta > 5:
        parts.append(
            f"LLM extraction covers {coverage_delta:.1f}% more agenda items."
        )
    elif coverage_delta < -5:
        parts.append(
            f"Keyword extraction covers {-coverage_delta:.1f}% more agenda items. "
            "Consider tuning LLM prompts."
        )
    else:
        parts.append("Both methods have similar agenda coverage.")

    if actor_count > 3:
        parts.append(
            f"LLM extraction identified {actor_count} unique actors, "
            "enabling speaker-level analysis."
        )

    if st_total > kw_total * 1.5:
        parts.append(
            "LLM extraction produces significantly more events. "
            "May need confidence filtering to reduce noise."
        )
    elif st_total < kw_total * 0.5:
        parts.append(
            "LLM extraction produces fewer events than keywords. "
            "Consider lowering confidence threshold or expanding prompts."
        )

    if not parts:
        parts.append("Both methods perform comparably for this meeting.")

    return " ".join(parts)
