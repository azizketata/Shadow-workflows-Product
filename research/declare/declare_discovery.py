"""Auto-discover Declare constraints from meeting event logs.

Uses pm4py's Declare discovery to find temporal constraints that hold
in the observed event data, then compares discovered constraints against
the normative Robert's Rules constraints to identify:
- Confirmed: normative constraints that the meeting actually followed
- Violated:  normative constraints the meeting broke
- Emergent:  constraints discovered in data but not in the normative set
"""

import sys
import os
from collections import defaultdict

import pandas as pd
import pm4py
from datetime import datetime, timedelta

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.time_utils import ts_to_seconds
from research.declare.roberts_rules import (
    ROBERTS_RULES_CONSTRAINTS,
    constraints_to_pm4py_format,
)


def _prepare_log(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a DataFrame for pm4py Declare operations."""
    log_df = df.copy()

    if "case:concept:name" not in log_df.columns:
        log_df["case:concept:name"] = "Meeting_1"

    if "concept:name" not in log_df.columns:
        if "mapped_activity" in log_df.columns:
            log_df["concept:name"] = log_df["mapped_activity"]
        elif "activity_name" in log_df.columns:
            log_df["concept:name"] = log_df["activity_name"]
        else:
            raise ValueError("No activity column found in DataFrame.")

    if "time:timestamp" not in log_df.columns:
        if "timestamp" in log_df.columns:
            def _parse(t_str):
                secs = ts_to_seconds(t_str)
                return datetime(2023, 1, 1) + timedelta(seconds=secs)
            log_df["time:timestamp"] = log_df["timestamp"].apply(_parse)
        else:
            log_df["time:timestamp"] = [
                datetime(2023, 1, 1) + timedelta(seconds=i)
                for i in range(len(log_df))
            ]

    return log_df


def discover_meeting_constraints(
    event_log_df: pd.DataFrame,
    min_support: float = 0.8,
) -> dict:
    """Auto-discover Declare constraints from a meeting event log.

    Uses pm4py's Declare discovery to find temporal patterns that hold
    with at least ``min_support`` confidence in the observed data.

    Parameters
    ----------
    event_log_df : pd.DataFrame
        Mapped event log with standard columns.
    min_support : float
        Minimum support threshold (0.0 to 1.0). Constraints holding in
        fewer than this fraction of traces are discarded. For single-trace
        meeting logs, use the default 0.8 or higher.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``discovered_model``: The pm4py Declare model object.
        - ``constraints``: List of discovered constraint dicts, each with
          ``template``, ``activities``, ``support``.
        - ``activity_count``: Number of unique activities in the log.
    """
    if event_log_df is None or event_log_df.empty:
        return {
            "discovered_model": None,
            "constraints": [],
            "activity_count": 0,
        }

    log_df = _prepare_log(event_log_df)
    unique_activities = log_df["concept:name"].nunique()

    try:
        declare_model = pm4py.discover_declare(log_df)
    except Exception as e:
        print(f"[Declare] Discovery failed: {e}")
        return {
            "discovered_model": None,
            "constraints": [],
            "activity_count": unique_activities,
        }

    # Parse the discovered model into a structured list
    parsed_constraints = _parse_declare_model(declare_model, min_support)

    return {
        "discovered_model": declare_model,
        "constraints": parsed_constraints,
        "activity_count": unique_activities,
    }


def _parse_declare_model(declare_model, min_support: float) -> list:
    """Parse a pm4py Declare model into a list of constraint dicts.

    The pm4py Declare model is a dictionary mapping template names
    to dictionaries of activity tuples and their support/confidence values.
    """
    constraints = []

    if declare_model is None:
        return constraints

    # pm4py Declare model is a dict: {template_str: {(act_tuple): confidence, ...}}
    if isinstance(declare_model, dict):
        for template_key, activity_map in declare_model.items():
            if not isinstance(activity_map, dict):
                continue
            for activity_tuple, confidence in activity_map.items():
                # confidence may be a float or a dict with 'support' key
                if isinstance(confidence, dict):
                    support = confidence.get("support", confidence.get("confidence", 0.0))
                elif isinstance(confidence, (int, float)):
                    support = float(confidence)
                else:
                    support = 0.0

                if support < min_support:
                    continue

                # Normalize activity_tuple
                if isinstance(activity_tuple, str):
                    activities = (activity_tuple,)
                elif isinstance(activity_tuple, tuple):
                    activities = activity_tuple
                else:
                    activities = (str(activity_tuple),)

                constraints.append({
                    "template": str(template_key),
                    "activities": activities,
                    "support": round(support, 4),
                })

    return constraints


def compare_discovered_vs_normative(
    discovered_constraints: list,
    normative_constraints: list = None,
) -> dict:
    """Compare discovered constraints against normative Robert's Rules.

    Parameters
    ----------
    discovered_constraints : list[dict]
        Output from ``discover_meeting_constraints()["constraints"]``.
    normative_constraints : list[tuple], optional
        List of ``(template, act_A, act_B, description)`` tuples.
        Defaults to ``ROBERTS_RULES_CONSTRAINTS``.

    Returns
    -------
    dict
        - ``confirmed``: Normative constraints also found in discovered data.
        - ``violated``: Normative constraints NOT found (potential violations).
        - ``emergent``: Discovered constraints NOT in normative set (emergent patterns).
        - ``counts``: Summary counts.
    """
    if normative_constraints is None:
        normative_constraints = ROBERTS_RULES_CONSTRAINTS

    # Build a lookup of discovered constraints as (template, activities) keys
    discovered_lookup = set()
    for dc in discovered_constraints:
        template = dc["template"].lower().replace(" ", "_")
        acts = dc["activities"]
        discovered_lookup.add((template, acts))
        # Also add with normalized template names (pm4py may use different naming)
        # e.g., "Precedence" -> "precedence"
        discovered_lookup.add((dc["template"].lower(), acts))

    # Build normative lookup for emergent comparison
    normative_lookup = set()
    for template, act_a, act_b, _desc in normative_constraints:
        if act_b is not None:
            normative_lookup.add((template.lower(), (act_a, act_b)))
        else:
            normative_lookup.add((template.lower(), (act_a,)))

    confirmed = []
    violated = []

    for template, act_a, act_b, desc in normative_constraints:
        norm_key_binary = (template.lower(), (act_a, act_b)) if act_b else None
        norm_key_unary = (template.lower(), (act_a,))

        found = False
        # Check if discovered constraints match
        for dk in discovered_lookup:
            d_template, d_acts = dk
            if d_template == template.lower():
                if act_b is None:
                    # Unary: just match act_a
                    if len(d_acts) >= 1 and d_acts[0] == act_a:
                        found = True
                        break
                else:
                    # Binary: match (act_a, act_b)
                    if len(d_acts) >= 2 and d_acts[0] == act_a and d_acts[1] == act_b:
                        found = True
                        break

        entry = {
            "template": template,
            "activity_a": act_a,
            "activity_b": act_b,
            "description": desc,
        }

        if found:
            confirmed.append(entry)
        else:
            violated.append(entry)

    # Find emergent constraints (discovered but not in normative set)
    emergent = []
    for dc in discovered_constraints:
        template = dc["template"].lower().replace(" ", "_")
        acts = dc["activities"]

        is_normative = False
        for norm_template, norm_acts in normative_lookup:
            if template == norm_template and acts == norm_acts:
                is_normative = True
                break

        if not is_normative:
            emergent.append({
                "template": dc["template"],
                "activities": acts,
                "support": dc["support"],
                "description": f"Emergent pattern: {dc['template']} on {acts}",
            })

    return {
        "confirmed": confirmed,
        "violated": violated,
        "emergent": emergent,
        "counts": {
            "normative_total": len(normative_constraints),
            "confirmed": len(confirmed),
            "violated": len(violated),
            "emergent": len(emergent),
            "compliance_ratio": (
                round(len(confirmed) / len(normative_constraints), 4)
                if normative_constraints
                else 0.0
            ),
        },
    }
