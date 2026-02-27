"""Centralized session state management with typed accessors."""

import streamlit as st
import pandas as pd


# All session state keys and their default factories/values
_DEFAULTS = {
    "debug_logs": list,
    "deviations": set,
    "accepted_deviations": set,
    "agenda_activities": list,
    "reference_bpmn": None,
    "last_alignments": None,
    "abstraction_cache": dict,
    "video_events": None,
    "abstracted_events": None,
    "mapped_events": None,
}


def init_state():
    """Initialize all session state keys with defaults (idempotent)."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default() if callable(default) else default


# ── Video Events ──────────────────────────────────────────────────────────────

def get_video_events():
    """Return raw video events DataFrame, or None."""
    return st.session_state.get("video_events")


def set_video_events(df):
    st.session_state["video_events"] = df


# ── Abstracted Events ────────────────────────────────────────────────────────

def get_abstracted_events():
    return st.session_state.get("abstracted_events")


def set_abstracted_events(df):
    st.session_state["abstracted_events"] = df


# ── Mapped Events ────────────────────────────────────────────────────────────

def get_mapped_events():
    return st.session_state.get("mapped_events")


def set_mapped_events(df):
    st.session_state["mapped_events"] = df


# ── Reference BPMN ───────────────────────────────────────────────────────────

def get_reference_bpmn():
    return st.session_state.get("reference_bpmn")


def set_reference_bpmn(bpmn_obj):
    st.session_state["reference_bpmn"] = bpmn_obj


# ── Agenda Activities ────────────────────────────────────────────────────────

def get_agenda_activities() -> list:
    return st.session_state.get("agenda_activities", [])


def set_agenda_activities(activities: list):
    st.session_state["agenda_activities"] = activities


# ── Alignments ───────────────────────────────────────────────────────────────

def get_alignments():
    return st.session_state.get("last_alignments")


def set_alignments(alignments):
    st.session_state["last_alignments"] = alignments


# ── Abstraction Cache ────────────────────────────────────────────────────────

def get_abstraction_cache() -> dict:
    return st.session_state.get("abstraction_cache", {})


# ── Deviations / Governance ──────────────────────────────────────────────────

def get_deviations() -> set:
    return st.session_state.get("deviations", set())


def add_deviation(dev: str):
    st.session_state.setdefault("deviations", set()).add(dev)


def get_accepted_deviations() -> set:
    return st.session_state.get("accepted_deviations", set())


def accept_deviation(dev: str):
    """Move a deviation from pending to accepted."""
    st.session_state.setdefault("accepted_deviations", set()).add(dev)
    st.session_state.get("deviations", set()).discard(dev)


# ── Debug Logs ───────────────────────────────────────────────────────────────

def get_debug_logs() -> list:
    return st.session_state.get("debug_logs", [])


def append_debug_log(message: str):
    logs = st.session_state.setdefault("debug_logs", [])
    logs.append(message)
    if len(logs) > 500:
        st.session_state["debug_logs"] = logs[-500:]
