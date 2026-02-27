"""Procedural Compliance Report Card — letter-grade summary of meeting conformance."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pipeline.time_utils import ts_to_seconds


# ---------------------------------------------------------------------------
# Grade helpers
# ---------------------------------------------------------------------------

_GRADE_RANGES = [
    (80, "A"), (65, "B"), (50, "C"), (35, "D"), (0, "F"),
]

_GRADE_COLORS = {
    "A": "#2e7d32", "B": "#558b2f", "C": "#f9a825", "D": "#e65100", "F": "#c62828",
}


def _letter_grade(pct: float) -> str:
    """Return letter grade with +/- modifier.

    Grade bands: A >=80, B >=65, C >=50, D >=35, F <35.
    Top third of each band gets +, bottom third gets -.
    """
    band_map = {"A": (80, 100), "B": (65, 80), "C": (50, 65), "D": (35, 50), "F": (0, 35)}

    letter = "F"
    for floor, ltr in _GRADE_RANGES:
        if pct >= floor:
            letter = ltr
            break

    if letter == "F":
        return "F"  # no F+ / F-

    lo, hi = band_map[letter]
    third = (hi - lo) / 3
    offset = pct - lo
    if offset >= 2 * third:
        return f"{letter}+"
    if offset < third:
        return f"{letter}-"
    return letter


def _base_letter(grade: str) -> str:
    return grade[0]


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_report_card(
    fitness_dedup: float,
    compliance_info: dict,
    alignments: list,
    mapped_events: pd.DataFrame,
    agenda_activities: list[str],
) -> dict:
    """Build a structured report-card dict from compliance artifacts.

    Parameters
    ----------
    fitness_dedup : float  (0-1 scale)
    compliance_info : dict  activity_name -> "executed"|"skipped"|"deviation"|"accepted"
    alignments : list of alignment dicts from PM4Py
    mapped_events : DataFrame with columns timestamp, activity_name, mapped_activity, source, original_text
    agenda_activities : ordered list of expected agenda labels

    Returns
    -------
    dict with keys: grade, fitness_pct, checklist, order_violations, summary_text
    """
    fitness_pct = round(fitness_dedup * 100, 1)
    grade = _letter_grade(fitness_pct)

    # --- Checklist ----------------------------------------------------------
    checklist: list[dict] = []
    for act in agenda_activities:
        status_raw = compliance_info.get(act, "skipped")
        if status_raw in ("executed", "accepted"):
            status = "held"
        elif status_raw == "deviation":
            status = "deviation"
        else:
            status = "skipped"

        # Count matching events
        count = 0
        if mapped_events is not None and not mapped_events.empty:
            col = "mapped_activity" if "mapped_activity" in mapped_events.columns else "activity_name"
            count = int((mapped_events[col] == act).sum())

        detail = f"{count} event(s) recorded" if status == "held" else (
            f"Deviation detected ({count} event(s))" if status == "deviation" else "No matching events"
        )
        checklist.append({"item": act, "status": status, "detail": detail})

    # --- Order violations ---------------------------------------------------
    order_violations: list[dict] = []
    if mapped_events is not None and not mapped_events.empty:
        col = "mapped_activity" if "mapped_activity" in mapped_events.columns else "activity_name"
        ts_col = "timestamp" if "timestamp" in mapped_events.columns else mapped_events.columns[0]

        # First occurrence of each agenda item in actual data
        first_occ: dict[str, int] = {}
        for _, row in mapped_events.iterrows():
            act = row.get(col)
            if act in agenda_activities and act not in first_occ:
                first_occ[act] = ts_to_seconds(row.get(ts_col, 0))

        # Build actual order (sorted by first-occurrence time)
        actual_order = sorted(first_occ.keys(), key=lambda a: first_occ[a])

        # Compare against expected order (only items that appeared)
        expected_sub = [a for a in agenda_activities if a in first_occ]
        for exp_pos, act in enumerate(expected_sub):
            if act in actual_order:
                act_pos = actual_order.index(act)
                if act_pos != exp_pos:
                    order_violations.append({
                        "expected": act,
                        "actual_position": act_pos + 1,
                        "expected_position": exp_pos + 1,
                    })

    # --- Summary text -------------------------------------------------------
    held = sum(1 for c in checklist if c["status"] == "held")
    skipped = sum(1 for c in checklist if c["status"] == "skipped")
    deviated = sum(1 for c in checklist if c["status"] == "deviation")
    total = len(checklist)

    parts = [f"Overall grade: {grade} ({fitness_pct}% dedup fitness)."]
    parts.append(f"{held}/{total} agenda items held, {skipped} skipped, {deviated} deviation(s).")
    if order_violations:
        parts.append(f"{len(order_violations)} order violation(s) detected.")
    else:
        parts.append("Agenda items followed the expected order.")

    return {
        "grade": grade,
        "fitness_pct": fitness_pct,
        "checklist": checklist,
        "order_violations": order_violations,
        "summary_text": " ".join(parts),
    }


# ---------------------------------------------------------------------------
# Streamlit renderer
# ---------------------------------------------------------------------------

def render_report_card(card: dict) -> None:
    """Render the report card in a Streamlit app."""
    grade = card["grade"]
    color = _GRADE_COLORS.get(_base_letter(grade), "#555555")

    # --- Letter grade banner ------------------------------------------------
    st.markdown(
        f"<div style='text-align:center;padding:18px 0 10px 0;'>"
        f"<span style='font-size:72px;font-weight:bold;color:{color};'>{grade}</span>"
        f"<br><span style='font-size:18px;color:#888;'>"
        f"Procedural Fitness: {card['fitness_pct']}%</span></div>",
        unsafe_allow_html=True,
    )

    # --- Summary ------------------------------------------------------------
    st.info(card["summary_text"])

    # --- Checklist ----------------------------------------------------------
    st.subheader("Agenda Checklist")
    for entry in card["checklist"]:
        icon = {"held": ":white_check_mark:", "skipped": ":x:", "deviation": ":warning:"}.get(
            entry["status"], ":grey_question:"
        )
        st.markdown(f"{icon} **{entry['item']}** -- {entry['detail']}")

    # --- Order violations ---------------------------------------------------
    if card["order_violations"]:
        st.subheader("Order Violations")
        for v in card["order_violations"]:
            st.markdown(
                f"- **{v['expected']}**: expected position {v['expected_position']}, "
                f"actual position {v['actual_position']}"
            )
    else:
        st.success("No order violations detected.")
