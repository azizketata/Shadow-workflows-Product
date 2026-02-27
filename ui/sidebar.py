"""Sidebar configuration widgets.  Returns a config dict consumed by the orchestrator."""

import streamlit as st
import time
import os
import tempfile
import pm4py

from pipeline.state import (
    get_abstracted_events, get_deviations, get_accepted_deviations,
    accept_deviation, get_debug_logs,
)
from bpmn_gen import convert_to_event_log


def render_sidebar() -> dict:
    """Render all sidebar widgets and return a flat config dict."""
    config = {}

    with st.sidebar:
        st.markdown("### Configuration")

        # ── API Key ───────────────────────────────────────────────────────────
        config["api_key"] = st.text_input(
            "OpenAI API Key", type="password",
            help="Required for agenda BPMN generation and LLM abstraction.",
        )
        if config["api_key"]:
            st.success("API key provided", icon="🔑")
        else:
            st.info("Enter your OpenAI API key to enable LLM features.", icon="🔑")

        st.divider()

        # ── Input Data ────────────────────────────────────────────────────────
        st.markdown("**Input Data**")
        config["uploaded_video"] = st.file_uploader("Meeting Video", type=["mp4"])
        config["agenda_text"] = st.text_area(
            "Meeting Agenda", height=160,
            help="Paste your meeting agenda here. Each line or numbered item becomes a BPMN task.",
        )

        st.divider()

        # ── Whisper ───────────────────────────────────────────────────────────
        with st.expander("Transcription (Whisper)", expanded=False):
            config["use_local_whisper"] = st.toggle("Use local Whisper (no API cost)", value=True)
            config["local_whisper_model"] = st.selectbox(
                "Local Whisper model", ["tiny", "base", "small", "medium"],
                index=2, help="Larger models are slower but more accurate.",
            )

        # ── Abstraction ───────────────────────────────────────────────────────
        with st.expander("LLM Abstraction", expanded=False):
            st.caption("Classifies sliding windows of transcript events into agenda labels using an LLM.")
            config["enable_abstraction"] = st.toggle("Enable LLM abstraction", value=True)
            config["openai_abstraction_model"] = st.selectbox(
                "LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"], index=0,
            )
            config["window_seconds"] = st.number_input(
                "Window size (seconds)", min_value=10, max_value=300, value=60, step=10,
                help="Size of sliding window. 60s is optimal for council meetings.",
            )
            config["overlap_ratio"] = st.slider(
                "Window overlap", min_value=0.0, max_value=0.9, value=0.5, step=0.1,
                help="0.5 is optimal. Higher = more redundancy.",
            )
            config["min_events_per_window"] = st.number_input(
                "Min events / window", min_value=1, max_value=50, value=2, step=1,
                help="Set to 2 to catch brief formal items (Roll Call, Pledge).",
            )
            config["min_label_support"] = st.number_input(
                "Min label support", min_value=1, max_value=20, value=1, step=1,
                help="Set to 1 to keep single-occurrence formal activities.",
            )
            config["shadow_min_ratio"] = st.slider(
                "Shadow min ratio", min_value=0.0, max_value=1.0, value=0.15, step=0.05,
            )
            config["max_windows_per_run"] = st.number_input(
                "Max windows / run", min_value=10, max_value=500, value=150, step=10,
            )
            config["openai_timeout"] = st.number_input(
                "LLM timeout (sec)", min_value=20, max_value=180, value=60, step=10,
            )
            config["abstraction_model"] = st.text_input("Ollama model (local fallback)", value="mistral")

        # ── SBERT Mapping ─────────────────────────────────────────────────────
        with st.expander("SBERT Mapping", expanded=False):
            st.caption("Sentence-BERT maps events to agenda items via cosine similarity.")
            config["sbert_model_choice"] = st.selectbox(
                "SBERT model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"], index=0,
                help="MiniLM is fast with better coverage; MPNet is slower with higher precision.",
            )
            config["sbert_threshold"] = st.slider(
                "Similarity threshold", min_value=0.10, max_value=0.70, value=0.35, step=0.05,
                help="Lower = more lenient matching. 0.35 is optimal for LLM abstraction; 0.20 for LLM-free.",
            )

        # ── Visual Detection ──────────────────────────────────────────────────
        with st.expander("Visual Detection", expanded=False):
            from video_processor import _RTMLIB_AVAILABLE
            if _RTMLIB_AVAILABLE:
                st.success("RTMPose (rtmlib) — available", icon="✅")
            else:
                st.warning("rtmlib not installed — pose detection disabled", icon="⚠️")
            st.caption("OpenCV MOG2 motion detection — always available")
            st.caption("Detects: hand raises (voting) + speaker activity (motion)")

        st.divider()

        # ── Debug ─────────────────────────────────────────────────────────────
        config["debug_enabled"] = st.checkbox("Enable debug logs", value=False)

        # NOTE: Export and Governance sections rendered by render_sidebar_dynamic()
        if config["debug_enabled"]:
            with st.expander("Debug Logs", expanded=False):
                logs = get_debug_logs()
                if logs:
                    st.code("\n".join(logs[-200:]))
                else:
                    st.write("No debug logs yet.")

    return config


def render_sidebar_dynamic():
    """Render sidebar sections that depend on post-processing session state.

    Must be called AFTER main-content processing so session state keys exist.
    """
    with st.sidebar:
        st.divider()

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown("**Export**")
        abstracted = get_abstracted_events()
        if abstracted is not None and hasattr(abstracted, "empty") and not abstracted.empty:
            csv_bytes = abstracted.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Abstracted CSV", data=csv_bytes,
                file_name="abstracted_log.csv", mime="text/csv",
            )
            temp_xes = None
            try:
                temp_xes = tempfile.NamedTemporaryFile(delete=False, suffix=".xes")
                temp_xes.close()
                log_df = convert_to_event_log(abstracted)
                if log_df is not None:
                    pm4py.write_xes(log_df, temp_xes.name)
                    with open(temp_xes.name, "rb") as f:
                        xes_bytes = f.read()
                    st.download_button(
                        "Download Abstracted XES", data=xes_bytes,
                        file_name="abstracted_log.xes", mime="application/xml",
                    )
            except Exception as e:
                st.warning(f"XES export failed: {e}")
            finally:
                if temp_xes is not None and os.path.exists(temp_xes.name):
                    try:
                        os.unlink(temp_xes.name)
                    except Exception:
                        pass
        else:
            st.caption("Process a video to enable exports.")

        st.divider()

        # ── Governance ────────────────────────────────────────────────────────
        st.markdown("**Governance: Shadow Activities**")
        deviations = get_deviations()
        if deviations:
            st.warning(f"{len(deviations)} shadow activities detected")
            for dev in list(deviations):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f'<div class="gov-card">{dev}</div>', unsafe_allow_html=True)
                if c2.button("Accept", key=f"acc_{dev}", help="Accept as formal variant"):
                    accept_deviation(dev)
                    st.rerun()
        else:
            st.caption("No shadow activities detected yet.")

        accepted = get_accepted_deviations()
        if accepted:
            st.success("Accepted Variants")
            for dev in accepted:
                st.markdown(f'<div class="gov-accepted">{dev}</div>', unsafe_allow_html=True)
