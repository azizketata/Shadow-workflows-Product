"""POWL discovery for shadow workflow analysis.

Discovers Partially Ordered Workflow Language models from mapped meeting events,
enabling non-block-structured representation of shadow workflows.

POWL models capture partial orderings that BPMN (block-structured) cannot
represent, making them ideal for analyzing how shadow/deviation activities
interleave with formal agenda items during meetings.

Requires pm4py >= 2.7.11 for discover_powl support.
"""

import sys
import os
import pandas as pd
import pm4py
from datetime import datetime, timedelta

# Allow imports from project root when running from research/ subdirectory
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bpmn_gen import convert_to_event_log
from pipeline.time_utils import ts_to_seconds


def _prepare_event_log(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a DataFrame for pm4py consumption.

    Ensures required columns exist: case:concept:name, concept:name,
    time:timestamp.  Operates on a copy so the caller's frame is untouched.
    """
    log_df = df.copy()

    if "case:concept:name" not in log_df.columns:
        log_df["case:concept:name"] = "Meeting_1"

    if "concept:name" not in log_df.columns:
        if "activity_name" in log_df.columns:
            log_df["concept:name"] = log_df["activity_name"]
        elif "mapped_activity" in log_df.columns:
            log_df["concept:name"] = log_df["mapped_activity"]
        else:
            raise ValueError(
                "DataFrame must contain 'concept:name', 'activity_name', "
                "or 'mapped_activity' column."
            )

    if "time:timestamp" not in log_df.columns:
        if "timestamp" in log_df.columns:
            def _parse(t_str):
                secs = ts_to_seconds(t_str)
                return datetime(2023, 1, 1) + timedelta(seconds=secs)

            log_df["time:timestamp"] = log_df["timestamp"].apply(_parse)
        else:
            # Fallback: generate monotonic timestamps from row order
            log_df["time:timestamp"] = [
                datetime(2023, 1, 1) + timedelta(seconds=i)
                for i in range(len(log_df))
            ]

    return log_df


def discover_shadow_powl(
    mapped_events_df: pd.DataFrame,
    frequency_threshold: float = 0.0,
):
    """Discover POWL model from shadow/deviation events only.

    Filters events where ``mapped_activity`` starts with ``"Deviation:"``,
    uses the original ``activity_name`` as the concept:name (the real label),
    and discovers a POWL model showing shadow workflow structure.

    Parameters
    ----------
    mapped_events_df : pd.DataFrame
        DataFrame with at least ``mapped_activity`` and ``activity_name``
        columns, plus ``timestamp`` for ordering.
    frequency_threshold : float
        Filtering weight factor for POWL discovery (0.0 keeps all behaviour).
        Mapped to pm4py's ``filtering_weight_factor`` parameter.

    Returns
    -------
    tuple
        ``(powl_model, shadow_event_count)`` or ``(None, 0)`` if there are
        fewer than 2 shadow events (insufficient for model discovery).
    """
    if mapped_events_df is None or mapped_events_df.empty:
        return None, 0

    if "mapped_activity" not in mapped_events_df.columns:
        return None, 0

    shadow_mask = mapped_events_df["mapped_activity"].str.startswith(
        "Deviation:", na=False
    )
    shadow_df = mapped_events_df[shadow_mask].copy()

    if len(shadow_df) < 2:
        return None, len(shadow_df)

    # Use the original activity_name as concept:name (not "Deviation: ...")
    if "activity_name" in shadow_df.columns:
        shadow_df["concept:name"] = shadow_df["activity_name"]
    else:
        # Strip the "Deviation: " prefix to get a meaningful label
        shadow_df["concept:name"] = shadow_df["mapped_activity"].str.replace(
            r"^Deviation:\s*", "", regex=True
        )

    log_df = _prepare_event_log(shadow_df)

    try:
        powl_model = pm4py.discover_powl(
            log_df,
            filtering_weight_factor=frequency_threshold,
        )
        return powl_model, len(shadow_df)
    except Exception as e:
        print(f"[POWL] Shadow discovery failed: {e}")
        return None, len(shadow_df)


def discover_full_powl(
    mapped_events_df: pd.DataFrame,
    frequency_threshold: float = 0.0,
):
    """Discover POWL model from ALL events (formal + shadow interleaved).

    The partial ordering reveals how shadow activities interleave with
    formal ones.  Uses ``mapped_activity`` as concept:name for formal events
    and ``activity_name`` for shadow/deviation events.

    Parameters
    ----------
    mapped_events_df : pd.DataFrame
        DataFrame with ``mapped_activity``, ``activity_name``, ``timestamp``.
    frequency_threshold : float
        Filtering weight factor for POWL discovery.
        Mapped to pm4py's ``filtering_weight_factor`` parameter.

    Returns
    -------
    tuple
        ``(powl_model, event_count)`` or ``(None, 0)`` on failure.
    """
    if mapped_events_df is None or mapped_events_df.empty:
        return None, 0

    df = mapped_events_df.copy()

    # Build mixed labels: formal events keep mapped_activity,
    # shadow/deviation events use the original activity_name
    if "mapped_activity" in df.columns and "activity_name" in df.columns:
        is_deviation = df["mapped_activity"].str.startswith("Deviation:", na=False)
        df["concept:name"] = df["mapped_activity"]
        df.loc[is_deviation, "concept:name"] = df.loc[is_deviation, "activity_name"]
    elif "mapped_activity" in df.columns:
        df["concept:name"] = df["mapped_activity"]
    elif "activity_name" in df.columns:
        df["concept:name"] = df["activity_name"]
    else:
        return None, 0

    log_df = _prepare_event_log(df)

    if len(log_df) < 2:
        return None, len(log_df)

    try:
        powl_model = pm4py.discover_powl(
            log_df,
            filtering_weight_factor=frequency_threshold,
        )
        return powl_model, len(log_df)
    except Exception as e:
        print(f"[POWL] Full discovery failed: {e}")
        return None, len(log_df)


def compare_bpmn_vs_powl(
    mapped_events_df: pd.DataFrame,
    bpmn_model,
    frequency_threshold: float = 0.0,
) -> dict:
    """Compare fitness of BPMN (block-structured) vs POWL (partial order).

    Computes conformance fitness for both model types against the same
    event log and returns a comparison dictionary.

    Parameters
    ----------
    mapped_events_df : pd.DataFrame
        Mapped event log with standard columns.
    bpmn_model : pm4py.objects.bpmn.obj.BPMN
        Reference BPMN model (from agenda or Inductive Miner).
    frequency_threshold : float
        Frequency threshold for POWL discovery.

    Returns
    -------
    dict
        Keys: ``bpmn_fitness``, ``powl_fitness``, ``fitness_delta``,
        ``structural_comparison``, ``powl_model``, ``event_count``.
        Returns zeroed values on failure.
    """
    result = {
        "bpmn_fitness": 0.0,
        "powl_fitness": 0.0,
        "fitness_delta": 0.0,
        "structural_comparison": {},
        "powl_model": None,
        "event_count": 0,
    }

    if mapped_events_df is None or mapped_events_df.empty:
        return result

    # Prepare the event log once for both comparisons
    log_df = convert_to_event_log(mapped_events_df)
    if log_df is None or log_df.empty:
        return result

    result["event_count"] = len(log_df)

    # --- BPMN fitness (via Petri net alignment) ---
    if bpmn_model is not None:
        try:
            net, im, fm = pm4py.convert_to_petri_net(bpmn_model)
            bpmn_fitness_result = pm4py.fitness_token_based_replay(
                log_df, net, im, fm
            )
            result["bpmn_fitness"] = bpmn_fitness_result.get("log_fitness", 0.0)
        except Exception as e:
            print(f"[POWL] BPMN fitness computation failed: {e}")

    # --- POWL discovery + fitness ---
    powl_model, _ = discover_full_powl(
        mapped_events_df, frequency_threshold=frequency_threshold
    )
    result["powl_model"] = powl_model

    if powl_model is not None:
        try:
            # Convert POWL to Petri net for conformance checking
            powl_net, powl_im, powl_fm = pm4py.convert_to_petri_net(powl_model)
            powl_fitness_result = pm4py.fitness_token_based_replay(
                log_df, powl_net, powl_im, powl_fm
            )
            result["powl_fitness"] = powl_fitness_result.get("log_fitness", 0.0)
        except Exception as e:
            print(f"[POWL] POWL fitness computation failed: {e}")

    # --- Structural comparison ---
    result["fitness_delta"] = result["powl_fitness"] - result["bpmn_fitness"]
    result["structural_comparison"] = {
        "bpmn_type": "block-structured",
        "powl_type": "partial-order",
        "bpmn_fitness": round(result["bpmn_fitness"], 4),
        "powl_fitness": round(result["powl_fitness"], 4),
        "delta": round(result["fitness_delta"], 4),
        "powl_advantage": result["fitness_delta"] > 0.01,
        "interpretation": (
            "POWL captures interleaving behaviour that BPMN cannot represent"
            if result["fitness_delta"] > 0.01
            else "BPMN and POWL show similar fitness; process may be block-structured"
        ),
    }

    return result


def visualize_powl(powl_model, output_path: str):
    """Save POWL model visualization to a file.

    Parameters
    ----------
    powl_model : pm4py POWL model
        The discovered POWL model to visualize.
    output_path : str
        File path for the output image (e.g., ``"shadow_powl.png"``).
        The format is inferred from the file extension.

    Raises
    ------
    ValueError
        If powl_model is None.
    """
    if powl_model is None:
        raise ValueError("Cannot visualize a None POWL model.")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    pm4py.save_vis_powl(powl_model, output_path)
