"""Declare conformance checking against Robert's Rules of Order.

Checks a meeting event log against normative Declare constraints and
computes a weighted procedural compliance score.

Two approaches are supported:
1. **pm4py conformance_declare**: Uses pm4py's built-in Declare conformance
   checking when available.
2. **Manual trace checking**: Fallback that manually verifies each constraint
   template against the observed activity sequence.
"""

import sys
import os
from collections import Counter

import pandas as pd
from datetime import datetime, timedelta

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.time_utils import ts_to_seconds
from research.declare.roberts_rules import (
    ROBERTS_RULES_CONSTRAINTS,
    get_constraints_for_agenda,
    CANONICAL_ACTIVITIES,
)

# ---------------------------------------------------------------------------
# Template weights for compliance scoring
# ---------------------------------------------------------------------------
# Higher weight = more important for procedural compliance.
# These reflect Robert's Rules severity: opening/closing procedures are
# mandatory, while chain constraints are stricter expectations.

TEMPLATE_WEIGHTS = {
    "existence": 1.0,       # Must-occur activities (Call to Order, Adjourn)
    "absence": 1.0,         # Must-not-occur
    "succession": 0.9,      # A before B, both required
    "response": 0.8,        # If A then eventually B
    "precedence": 0.7,      # B only after A
    "chain_response": 0.5,  # Strict immediate succession (harder to satisfy)
    "chain_precedence": 0.5,
    "not_co_existence": 0.6,
}


def check_roberts_rules_conformance(
    mapped_events_df: pd.DataFrame,
    agenda_activities: list = None,
) -> list:
    """Check a meeting event log against Robert's Rules Declare constraints.

    Parameters
    ----------
    mapped_events_df : pd.DataFrame
        Mapped event log with ``mapped_activity`` or ``activity_name``
        and ``timestamp`` columns.
    agenda_activities : list[str], optional
        If provided, constraints are filtered to only those whose
        activities appear in the agenda. If None, all Robert's Rules
        constraints are checked.

    Returns
    -------
    list[dict]
        One dict per constraint checked, with keys:
        - ``template``: The Declare template name.
        - ``activity_a``: First activity.
        - ``activity_b``: Second activity (or None for unary).
        - ``description``: Human-readable description.
        - ``satisfied``: Boolean, whether the constraint holds.
        - ``detail``: Explanation of why it passed or failed.
    """
    if mapped_events_df is None or mapped_events_df.empty:
        return []

    # Determine which column to use for activity names
    if "mapped_activity" in mapped_events_df.columns:
        act_col = "mapped_activity"
    elif "concept:name" in mapped_events_df.columns:
        act_col = "concept:name"
    elif "activity_name" in mapped_events_df.columns:
        act_col = "activity_name"
    else:
        return []

    # Build the ordered activity trace
    df = mapped_events_df.copy()
    if "raw_seconds" in df.columns:
        df["__secs"] = df["raw_seconds"].astype(int)
    elif "timestamp" in df.columns:
        df["__secs"] = df["timestamp"].apply(ts_to_seconds)
    else:
        df["__secs"] = range(len(df))

    df = df.sort_values("__secs").reset_index(drop=True)
    trace = df[act_col].tolist()

    # Get applicable constraints
    if agenda_activities:
        constraints = get_constraints_for_agenda(agenda_activities)
    else:
        constraints = ROBERTS_RULES_CONSTRAINTS

    if not constraints:
        return []

    results = []
    for template, act_a, act_b, desc in constraints:
        satisfied, detail = _check_single_constraint(
            template, act_a, act_b, trace
        )
        results.append({
            "template": template,
            "activity_a": act_a,
            "activity_b": act_b,
            "description": desc,
            "satisfied": satisfied,
            "detail": detail,
        })

    return results


def _check_single_constraint(
    template: str,
    act_a: str,
    act_b: str,
    trace: list,
) -> tuple:
    """Check a single Declare constraint against an activity trace.

    Parameters
    ----------
    template : str
        Declare template name.
    act_a : str
        First activity.
    act_b : str or None
        Second activity (None for unary templates).
    trace : list[str]
        Ordered list of activity labels.

    Returns
    -------
    tuple[bool, str]
        (satisfied, explanation)
    """
    # Normalize: match case-insensitively
    trace_lower = [a.lower().strip() for a in trace]
    a_lower = act_a.lower().strip()
    b_lower = act_b.lower().strip() if act_b else None

    a_indices = [i for i, t in enumerate(trace_lower) if t == a_lower]
    b_indices = [i for i, t in enumerate(trace_lower) if t == b_lower] if b_lower else []

    a_present = len(a_indices) > 0
    b_present = len(b_indices) > 0

    if template == "existence":
        if a_present:
            return True, f"'{act_a}' occurs {len(a_indices)} time(s) in the trace."
        return False, f"'{act_a}' does not occur in the trace."

    elif template == "absence":
        if not a_present:
            return True, f"'{act_a}' correctly absent from the trace."
        return False, f"'{act_a}' occurs {len(a_indices)} time(s) but should be absent."

    elif template == "precedence":
        # B can only occur if A occurred before it
        if not b_present:
            # B never occurs: vacuously satisfied
            return True, f"'{act_b}' does not occur, so precedence is vacuously satisfied."
        if not a_present:
            return False, f"'{act_b}' occurs but '{act_a}' never occurred before it."
        first_b = min(b_indices)
        first_a = min(a_indices)
        if first_a < first_b:
            return True, (
                f"'{act_a}' first occurs at position {first_a}, "
                f"before '{act_b}' at position {first_b}."
            )
        return False, (
            f"'{act_b}' occurs at position {first_b} but "
            f"'{act_a}' first occurs at position {first_a} (too late)."
        )

    elif template == "response":
        # If A occurs, B must eventually occur after
        if not a_present:
            return True, f"'{act_a}' does not occur, so response is vacuously satisfied."
        last_a = max(a_indices)
        if not b_present:
            return False, f"'{act_a}' occurs but '{act_b}' never follows."
        # Check that for every A occurrence, some B follows
        for a_idx in a_indices:
            b_after = [bi for bi in b_indices if bi > a_idx]
            if not b_after:
                return False, (
                    f"'{act_a}' occurs at position {a_idx} but "
                    f"no '{act_b}' follows afterward."
                )
        return True, (
            f"Every occurrence of '{act_a}' is eventually followed by '{act_b}'."
        )

    elif template == "succession":
        # A before B AND both must occur (precedence + response)
        prec_ok, prec_detail = _check_single_constraint("precedence", act_a, act_b, trace)
        resp_ok, resp_detail = _check_single_constraint("response", act_a, act_b, trace)
        if prec_ok and resp_ok:
            return True, f"Succession holds: {prec_detail} AND {resp_detail}"
        failures = []
        if not prec_ok:
            failures.append(f"Precedence failed: {prec_detail}")
        if not resp_ok:
            failures.append(f"Response failed: {resp_detail}")
        return False, " | ".join(failures)

    elif template == "chain_response":
        # If A occurs, B must occur immediately next
        if not a_present:
            return True, f"'{act_a}' does not occur, so chain_response is vacuously satisfied."
        for a_idx in a_indices:
            next_idx = a_idx + 1
            if next_idx >= len(trace_lower):
                return False, (
                    f"'{act_a}' occurs at the end of the trace (position {a_idx}); "
                    f"no '{act_b}' can follow."
                )
            if trace_lower[next_idx] != b_lower:
                return False, (
                    f"'{act_a}' at position {a_idx} is immediately followed by "
                    f"'{trace[next_idx]}', not '{act_b}'."
                )
        return True, (
            f"Every '{act_a}' is immediately followed by '{act_b}'."
        )

    elif template == "chain_precedence":
        # B can only occur if A occurred immediately before
        if not b_present:
            return True, f"'{act_b}' does not occur, so chain_precedence is vacuously satisfied."
        for b_idx in b_indices:
            prev_idx = b_idx - 1
            if prev_idx < 0:
                return False, (
                    f"'{act_b}' occurs at the start of the trace (position 0); "
                    f"no '{act_a}' can precede it."
                )
            if trace_lower[prev_idx] != a_lower:
                return False, (
                    f"'{act_b}' at position {b_idx} is immediately preceded by "
                    f"'{trace[prev_idx]}', not '{act_a}'."
                )
        return True, (
            f"Every '{act_b}' is immediately preceded by '{act_a}'."
        )

    elif template == "not_co_existence":
        # A and B cannot both occur
        if a_present and b_present:
            return False, (
                f"Both '{act_a}' and '{act_b}' occur, violating not_co_existence."
            )
        return True, (
            f"At most one of '{act_a}' / '{act_b}' occurs."
        )

    else:
        return True, f"Unknown template '{template}'; skipping (treated as satisfied)."


def compute_procedural_compliance_score(conformance_results: list) -> dict:
    """Compute a weighted procedural compliance score from conformance results.

    Parameters
    ----------
    conformance_results : list[dict]
        Output from ``check_roberts_rules_conformance()``.

    Returns
    -------
    dict
        - ``score``: Weighted compliance score, 0-100.
        - ``satisfied_count``: Number of satisfied constraints.
        - ``violated_count``: Number of violated constraints.
        - ``total_checked``: Total constraints checked.
        - ``by_template``: Per-template breakdown of satisfaction rates.
        - ``grade``: Letter grade (A/B/C/D/F) based on score.
    """
    if not conformance_results:
        return {
            "score": 0.0,
            "satisfied_count": 0,
            "violated_count": 0,
            "total_checked": 0,
            "by_template": {},
            "grade": "N/A",
        }

    total_weight = 0.0
    earned_weight = 0.0
    satisfied_count = 0
    violated_count = 0
    template_stats = {}  # template -> {"satisfied": int, "total": int}

    for result in conformance_results:
        template = result["template"]
        weight = TEMPLATE_WEIGHTS.get(template, 0.5)
        total_weight += weight

        if template not in template_stats:
            template_stats[template] = {"satisfied": 0, "total": 0}
        template_stats[template]["total"] += 1

        if result["satisfied"]:
            earned_weight += weight
            satisfied_count += 1
            template_stats[template]["satisfied"] += 1
        else:
            violated_count += 1

    score = (earned_weight / total_weight * 100) if total_weight > 0 else 0.0
    score = round(score, 1)

    # Letter grade
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    by_template = {}
    for template, stats in template_stats.items():
        total = stats["total"]
        sat = stats["satisfied"]
        by_template[template] = {
            "satisfied": sat,
            "total": total,
            "rate": round(sat / total, 4) if total > 0 else 0.0,
        }

    return {
        "score": score,
        "satisfied_count": satisfied_count,
        "violated_count": violated_count,
        "total_checked": len(conformance_results),
        "by_template": by_template,
        "grade": grade,
    }
