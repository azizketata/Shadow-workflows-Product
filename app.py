import streamlit as st
import tempfile
import os
import pandas as pd
import pm4py
import time
from bpmn_gen import (
    generate_agenda_bpmn,
    convert_to_event_log,
    generate_discovered_bpmn,
    generate_colored_bpmn,
)
from video_processor import VideoProcessor, _RTMLIB_AVAILABLE
from compliance_engine import ComplianceEngine

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Process Twin",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

[data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header banner ────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: #ffffff;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -30%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(83,120,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    font-size: 0.95rem;
    opacity: 0.85;
    margin: 0;
    max-width: 600px;
}

/* ── Metric cards row ─────────────────────────────────── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e8ecf1;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6b7280;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #9ca3af;
    margin-top: 0.25rem;
}

/* colour classes */
.metric-card.green  .value { color: #059669; }
.metric-card.blue   .value { color: #2563eb; }
.metric-card.amber  .value { color: #d97706; }
.metric-card.red    .value { color: #dc2626; }
.metric-card.purple .value { color: #7c3aed; }

/* ── Status bar ───────────────────────────────────────── */
.status-bar {
    background: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    font-size: 0.88rem;
}
.status-bar .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.status-bar .dot.live { background: #059669; animation: pulse 1.5s infinite; }
.status-bar .dot.idle { background: #9ca3af; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Section headers ──────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    margin: 1.2rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Sidebar polish ───────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #fafbfc;
}
[data-testid="stSidebar"] hr {
    margin: 0.8rem 0;
    border-color: #e5e7eb;
}

/* ── Evidence panel ───────────────────────────────────── */
.evidence-item {
    background: #f8fafc;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.evidence-item strong {
    color: #1e40af;
}

/* ── Governance cards ─────────────────────────────────── */
.gov-card {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}
.gov-accepted {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}

/* ── Placeholder card ─────────────────────────────────── */
.placeholder-card {
    border: 2px dashed #cbd5e1;
    padding: 3rem 2rem;
    text-align: center;
    border-radius: 16px;
    color: #94a3b8;
    background: #f8fafc;
}
.placeholder-card h3 {
    color: #64748b;
    margin-bottom: 0.5rem;
}

/* ── hide default metric colours ──────────────────────── */
[data-testid="stMetricDelta"] { display: none; }

/* ── Reduce sidebar widget padding ────────────────────── */
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stNumberInput,
[data-testid="stSidebar"] .stSlider {
    margin-bottom: -0.3rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session State Initialisation ───────────────────────────────────────────────
_defaults = {
    "debug_logs": [],
    "deviations": set(),
    "accepted_deviations": set(),
    "agenda_activities": [],
    "reference_bpmn": None,
    "last_alignments": None,
    "abstraction_cache": {},
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Debug Logger ───────────────────────────────────────────────────────────────
def log_debug(message):
    if debug_enabled:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        st.session_state["debug_logs"].append(line)
        if len(st.session_state["debug_logs"]) > 500:
            st.session_state["debug_logs"] = st.session_state["debug_logs"][-500:]


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Configuration")

    # ── API Key ────────────────────────────────────────────────────────────────
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Required for agenda BPMN generation and LLM abstraction.",
    )
    if api_key:
        st.success("API key provided", icon="🔑")
    else:
        st.info("Enter your OpenAI API key to enable LLM features.", icon="🔑")

    st.divider()

    # ── Input Data ─────────────────────────────────────────────────────────────
    st.markdown("**Input Data**")
    uploaded_video = st.file_uploader("Meeting Video", type=["mp4"])
    agenda_text = st.text_area(
        "Meeting Agenda",
        height=160,
        help="Paste your meeting agenda here. Each line or numbered item becomes a BPMN task.",
    )

    st.divider()

    # ── Whisper ────────────────────────────────────────────────────────────────
    with st.expander("Transcription (Whisper)", expanded=False):
        use_local_whisper = st.toggle("Use local Whisper (no API cost)", value=True)
        local_whisper_model = st.selectbox(
            "Local Whisper model",
            ["tiny", "base", "small", "medium"],
            index=2,  # default: small
            help="Larger models are slower but more accurate.",
        )

    # ── Abstraction ────────────────────────────────────────────────────────────
    with st.expander("LLM Abstraction", expanded=False):
        st.caption("Classifies sliding windows of transcript events into agenda labels using an LLM.")
        enable_abstraction = st.toggle("Enable LLM abstraction", value=True)
        openai_abstraction_model = st.selectbox(
            "LLM Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"],
            index=0,
        )
        window_seconds = st.number_input(
            "Window size (seconds)", min_value=10, max_value=300, value=60, step=10,
            help="Size of sliding window. 60s is optimal for council meetings.",
        )
        overlap_ratio = st.slider(
            "Window overlap", min_value=0.0, max_value=0.9, value=0.5, step=0.1,
            help="0.5 is optimal. Higher = more redundancy.",
        )
        min_events_per_window = st.number_input(
            "Min events / window", min_value=1, max_value=50, value=2, step=1,
            help="Set to 2 to catch brief formal items (Roll Call, Pledge).",
        )
        min_label_support = st.number_input(
            "Min label support", min_value=1, max_value=20, value=1, step=1,
            help="Set to 1 to keep single-occurrence formal activities.",
        )
        shadow_min_ratio = st.slider(
            "Shadow min ratio", min_value=0.0, max_value=1.0, value=0.15, step=0.05,
        )
        max_windows_per_run = st.number_input(
            "Max windows / run", min_value=10, max_value=500, value=150, step=10,
        )
        openai_timeout = st.number_input(
            "LLM timeout (sec)", min_value=20, max_value=180, value=60, step=10,
        )
        abstraction_model = st.text_input("Ollama model (local fallback)", value="mistral")

    # ── SBERT Mapping ──────────────────────────────────────────────────────────
    with st.expander("SBERT Mapping", expanded=False):
        st.caption("Sentence-BERT maps events to agenda items via cosine similarity.")
        sbert_model_choice = st.selectbox(
            "SBERT model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            index=0,
            help="MiniLM is fast with better coverage; MPNet is slower with higher precision.",
        )
        sbert_threshold = st.slider(
            "Similarity threshold",
            min_value=0.10, max_value=0.70, value=0.35, step=0.05,
            help="Lower = more lenient matching. 0.35 is optimal for LLM abstraction; 0.20 for LLM-free.",
        )

    # ── Visual Detection ──────────────────────────────────────────────────────
    with st.expander("Visual Detection", expanded=False):
        if _RTMLIB_AVAILABLE:
            st.success("RTMPose (rtmlib) — available", icon="✅")
        else:
            st.warning("rtmlib not installed — pose detection disabled", icon="⚠️")
        st.caption("OpenCV MOG2 motion detection — always available")
        st.caption("Detects: hand raises (voting) + speaker activity (motion)")

    st.divider()

    # ── Debug ──────────────────────────────────────────────────────────────────
    debug_enabled = st.checkbox("Enable debug logs", value=False)

    # ── Export ─────────────────────────────────────────────────────────────────
    st.markdown("**Export**")
    abstracted_for_export = st.session_state.get("abstracted_events")
    if abstracted_for_export is not None and not isinstance(abstracted_for_export, type(None)):
        if hasattr(abstracted_for_export, "empty") and not abstracted_for_export.empty:
            csv_bytes = abstracted_for_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Abstracted CSV",
                data=csv_bytes,
                file_name="abstracted_log.csv",
                mime="text/csv",
            )
            temp_xes = None
            try:
                temp_xes = tempfile.NamedTemporaryFile(delete=False, suffix=".xes")
                temp_xes.close()
                log_df = convert_to_event_log(abstracted_for_export)
                if log_df is not None:
                    pm4py.write_xes(log_df, temp_xes.name)
                    with open(temp_xes.name, "rb") as f:
                        xes_bytes = f.read()
                    st.download_button(
                        "Download Abstracted XES",
                        data=xes_bytes,
                        file_name="abstracted_log.xes",
                        mime="application/xml",
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
    else:
        st.caption("Process a video to enable exports.")

    st.divider()

    # ── Governance ─────────────────────────────────────────────────────────────
    st.markdown("**Governance: Shadow Activities**")
    if st.session_state["deviations"]:
        st.warning(f"{len(st.session_state['deviations'])} shadow activities detected")
        for dev in list(st.session_state["deviations"]):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f'<div class="gov-card">{dev}</div>', unsafe_allow_html=True)
            if c2.button("Accept", key=f"acc_{dev}", help="Accept as formal variant"):
                st.session_state["accepted_deviations"].add(dev)
                st.session_state["deviations"].discard(dev)
                st.rerun()
    else:
        st.caption("No shadow activities detected yet.")

    if st.session_state["accepted_deviations"]:
        st.success("Accepted Variants")
        for dev in st.session_state["accepted_deviations"]:
            st.markdown(f'<div class="gov-accepted">{dev}</div>', unsafe_allow_html=True)

    # ── Debug Log Output ───────────────────────────────────────────────────────
    if debug_enabled:
        with st.expander("Debug Logs", expanded=False):
            if st.session_state["debug_logs"]:
                st.code("\n".join(st.session_state["debug_logs"][-200:]))
            else:
                st.write("No debug logs yet.")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPLIANCE ENGINE (cached per SBERT model choice)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_compliance_engine(model_name):
    return ComplianceEngine(sbert_model_name=model_name)


compliance_engine = load_compliance_engine(sbert_model_choice)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="hero-banner">
    <h1>Meeting Process Twin</h1>
    <p>Transform meeting recordings into BPMN compliance reports.
       Upload a video, paste the agenda, and watch conformance unfold in real time.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC CARDS (top-level)
# ══════════════════════════════════════════════════════════════════════════════
metric_placeholder = st.empty()


def render_metrics(
    fitness_raw_pct=0.0, fitness_dedup_pct=0.0,
    events_count=0, agenda_count=0, matched_count=0, shadow_count=0,
    visual_votes=0, visual_motion=0,
):
    """Render the top metric cards."""
    # colour logic — based on dedup fitness (primary metric)
    if events_count == 0:
        dedup_cls = "blue"
    elif fitness_dedup_pct >= 65:
        dedup_cls = "green"
    elif fitness_dedup_pct >= 40:
        dedup_cls = "amber"
    else:
        dedup_cls = "red"

    # raw fitness colour (usually lower, separate colouring)
    if events_count == 0:
        raw_cls = "blue"
    elif fitness_raw_pct >= 50:
        raw_cls = "green"
    elif fitness_raw_pct >= 25:
        raw_cls = "amber"
    else:
        raw_cls = "red"

    # visual events colour
    vis_total = visual_votes + visual_motion
    vis_cls = "green" if vis_total > 0 else "blue"

    metric_placeholder.markdown(
        f"""
<div class="metric-row">
    <div class="metric-card {dedup_cls}">
        <div class="label">Dedup Fitness (primary)</div>
        <div class="value">{fitness_dedup_pct:.1f}%</div>
        <div class="sub">First-occurrence trace</div>
    </div>
    <div class="metric-card {raw_cls}">
        <div class="label">Raw Fitness</div>
        <div class="value">{fitness_raw_pct:.1f}%</div>
        <div class="sub">All events (strict)</div>
    </div>
    <div class="metric-card blue">
        <div class="label">Events Detected</div>
        <div class="value">{events_count}</div>
        <div class="sub">Audio + Visual fused</div>
    </div>
    <div class="metric-card purple">
        <div class="label">Agenda Items</div>
        <div class="value">{agenda_count}</div>
        <div class="sub">Reference model tasks</div>
    </div>
    <div class="metric-card amber">
        <div class="label">Matched / Shadow</div>
        <div class="value">{matched_count} / {shadow_count}</div>
        <div class="sub">Formal vs informal</div>
    </div>
    <div class="metric-card {vis_cls}">
        <div class="label">Visual Events</div>
        <div class="value">{vis_total}</div>
        <div class="sub">{visual_votes} votes · {visual_motion} motion</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# Initial render
render_metrics()


# ══════════════════════════════════════════════════════════════════════════════
#  OVERLAY TOGGLE
# ══════════════════════════════════════════════════════════════════════════════
show_overlay = st.toggle("Show Shadow Workflow Overlay", value=False)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

# ── Placeholders inside columns (so we can update them later) ──────────────
with col1:
    st.markdown('<div class="section-header">📋 Reference: Agenda BPMN</div>', unsafe_allow_html=True)
    ref_bpmn_placeholder = st.empty()

with col2:
    st.markdown('<div class="section-header">🔍 Discovered: Real-time Video BPMN</div>', unsafe_allow_html=True)
    disc_bpmn_placeholder = st.empty()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — AGENDA BPMN (Left Column)
# ══════════════════════════════════════════════════════════════════════════════
if agenda_text:
    if not api_key:
        with col1:
            ref_bpmn_placeholder.warning("Enter your OpenAI API Key in the sidebar to generate the agenda BPMN.")
    else:
        if st.session_state["reference_bpmn"] is None:
            with col1:
                with st.spinner("Generating BPMN from agenda..."):
                    try:
                        bpmn_viz, activities, bpmn_obj = generate_agenda_bpmn(agenda_text, api_key)
                        if bpmn_viz and bpmn_obj:
                            st.session_state["agenda_activities"] = activities
                            st.session_state["reference_bpmn"] = bpmn_obj
                            log_debug(f"Agenda BPMN generated — {len(activities)} activities")
                        else:
                            ref_bpmn_placeholder.error("Failed to generate BPMN from agenda.")
                            log_debug("Agenda BPMN generation returned None.")
                    except Exception as e:
                        ref_bpmn_placeholder.error(f"Error: {e}")
else:
    with col1:
        ref_bpmn_placeholder.markdown(
            '<div class="placeholder-card"><h3>No Agenda</h3><p>Paste your meeting agenda in the sidebar.</p></div>',
            unsafe_allow_html=True,
        )


# ── Render reference BPMN (standard or overlay) ───────────────────────────────
def render_reference_bpmn():
    """Render the reference BPMN in col1, either colored (overlay) or plain."""
    bpmn_obj = st.session_state.get("reference_bpmn")
    if not bpmn_obj:
        return

    alignments = st.session_state.get("last_alignments")

    with col1:
        if show_overlay and alignments:
            colored_viz, compliance_info = generate_colored_bpmn(bpmn_obj, alignments)
            if colored_viz:
                ref_bpmn_placeholder.graphviz_chart(colored_viz, width='stretch')
                st.caption("🟢 Executed  |  ⬜ Skipped  |  🔶 Shadow (see sidebar)")
                if compliance_info:
                    with st.expander("Compliance Status per Agenda Item", expanded=False):
                        for act, status in compliance_info.items():
                            if status == "executed":
                                st.success(f"**{act}** — matched")
                            elif status == "skipped":
                                st.warning(f"**{act}** — not detected")
                            else:
                                st.info(f"**{act}** — {status}")
            else:
                ref_bpmn_placeholder.info("Overlay could not be generated.")
        elif show_overlay and not alignments:
            # Overlay requested but no alignment data yet — show plain + message
            from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
            params = {bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg"}
            gviz = bpmn_visualizer.apply(bpmn_obj, parameters=params)
            gviz.attr(rankdir="TB")
            ref_bpmn_placeholder.graphviz_chart(gviz, width='stretch')
            st.info("Process the video first to generate overlay data.")
        else:
            # Standard view
            from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
            params = {bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg"}
            gviz = bpmn_visualizer.apply(bpmn_obj, parameters=params)
            gviz.attr(rankdir="TB")
            ref_bpmn_placeholder.graphviz_chart(gviz, width='stretch')


render_reference_bpmn()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — VIDEO PROCESSING (Right Column)
# ══════════════════════════════════════════════════════════════════════════════
# Time conversion helpers
def _ts_to_sec(t_str):
    parts = str(t_str).split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        pass
    return 0


def _sec_to_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


with col2:
    if uploaded_video is not None:
        st.video(uploaded_video)

        if not api_key and not use_local_whisper:
            disc_bpmn_placeholder.warning(
                "Enter your OpenAI API Key or enable local Whisper in the sidebar."
            )
        else:
            # ── Process video (once) ───────────────────────────────────────────
            if "video_events" not in st.session_state:
                with st.spinner("Analysing audio and visual streams..."):
                    try:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(uploaded_video.read())
                        video_path = tfile.name
                        tfile.close()

                        processor = VideoProcessor(api_key or "local", debug=debug_enabled)
                        df = processor.process_video(
                            video_path,
                            use_local_whisper=use_local_whisper,
                            local_whisper_model=local_whisper_model,
                        )
                        st.session_state["video_events"] = df
                        log_debug(f"Video processing complete — {df.shape[0]} events")
                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                        log_debug(f"Video processing error: {e}")
                    finally:
                        if "video_path" in locals() and os.path.exists(video_path):
                            os.unlink(video_path)

            # ── Simulation Slider ──────────────────────────────────────────────
            if "video_events" in st.session_state:
                df_events = st.session_state["video_events"]

                max_seconds = 0
                if not df_events.empty:
                    max_seconds = _ts_to_sec(df_events.iloc[-1]["timestamp"])

                time_options = [_sec_to_ts(s) for s in range(0, max_seconds + 60, 10)]
                if not time_options:
                    time_options = ["00:00:00"]

                selected_time_str = st.select_slider(
                    "Jump to time",
                    options=time_options,
                    value=time_options[-1],  # default to end so full trace is visible
                    key="video_time_slider",
                )
                current_video_time = _ts_to_sec(selected_time_str)

                # Filter events up to selected time
                current_events = df_events[
                    df_events["timestamp"].apply(_ts_to_sec) <= current_video_time
                ]
                log_debug(f"Slider: {selected_time_str} — {len(current_events)} events in scope")

                # ── Phase 3: Semantic Abstraction ──────────────────────────────
                abstracted_events = current_events
                if (
                    enable_abstraction
                    and not current_events.empty
                    and st.session_state["agenda_activities"]
                    and api_key
                ):
                    try:
                        abstracted_events = compliance_engine.abstract_events_df(
                            df=current_events,
                            agenda_tasks=st.session_state["agenda_activities"],
                            window_seconds=window_seconds,
                            shadow_min_ratio=shadow_min_ratio,
                            model=abstraction_model,
                            overlap_ratio=overlap_ratio,
                            min_events_per_window=min_events_per_window,
                            min_label_support=min_label_support,
                            api_key=api_key,
                            openai_model=openai_abstraction_model,
                            openai_timeout=openai_timeout,
                            max_windows_per_run=max_windows_per_run,
                            cache=st.session_state["abstraction_cache"],
                            debug_callback=log_debug,
                        )
                        st.session_state["abstracted_events"] = abstracted_events
                        log_debug(f"Abstraction produced {len(abstracted_events)} events")
                    except Exception as e:
                        log_debug(f"Abstraction error: {e}")
                        st.warning(f"Abstraction failed — using raw events. {e}")
                        abstracted_events = current_events
                else:
                    st.session_state["abstracted_events"] = abstracted_events

                # ── Phase 4: Compliance Check ──────────────────────────────────
                fitness_score = 0.0
                fitness_dedup = 0.0
                alignments = []
                mapped_events = abstracted_events
                shadow_count = 0
                matched_count = 0

                if (
                    not abstracted_events.empty
                    and st.session_state["reference_bpmn"]
                    and st.session_state["agenda_activities"]
                ):
                    mapped_events = compliance_engine.map_events_to_agenda(
                        abstracted_events,
                        st.session_state["agenda_activities"],
                        threshold=sbert_threshold,
                    )
                    log_debug(f"Mapped events: {len(mapped_events)}")

                    # Count formal vs shadow
                    if "mapped_activity" in mapped_events.columns:
                        shadow_count = mapped_events["mapped_activity"].str.startswith("Deviation:", na=False).sum()
                        matched_count = len(mapped_events) - shadow_count

                    log_data_for_fitness = convert_to_event_log(mapped_events)
                    if log_data_for_fitness is not None:
                        # Raw fitness (all events)
                        compliance_result = compliance_engine.calculate_fitness(
                            st.session_state["reference_bpmn"],
                            log_data_for_fitness,
                        )
                        fitness_score = compliance_result.get("score", 0.0)
                        alignments = compliance_result.get("alignments", [])
                        st.session_state["last_alignments"] = alignments
                        log_debug(f"Raw fitness: {fitness_score:.4f}")

                        # Dedup fitness (first occurrence per agenda item)
                        try:
                            _act_col = "mapped_activity" if "mapped_activity" in mapped_events.columns else "activity_name"
                            _formal = mapped_events[
                                ~mapped_events[_act_col].str.startswith("Deviation:", na=False)
                            ].copy()
                            if not _formal.empty:
                                _first = _formal.groupby(_act_col).first().reset_index()
                                _first["__ts"] = _first["timestamp"].apply(_ts_to_sec)
                                _first = _first.sort_values("__ts").drop(columns=["__ts"])
                                _first["activity_name"] = _first[_act_col]
                                _dedup_log = convert_to_event_log(_first)
                                if _dedup_log is not None and not _dedup_log.empty:
                                    _dedup_result = compliance_engine.calculate_fitness(
                                        st.session_state["reference_bpmn"],
                                        _dedup_log,
                                    )
                                    fitness_dedup = _dedup_result.get("score", 0.0)
                                    log_debug(f"Dedup fitness: {fitness_dedup:.4f} (trace length: {len(_first)})")
                        except Exception as e:
                            log_debug(f"Dedup fitness error: {e}")

                        # Detect shadow activities for governance
                        for align in alignments:
                            for log_move, model_move in align["alignment"]:
                                if (model_move is None or model_move == ">>") and log_move is not None:
                                    if log_move not in st.session_state["accepted_deviations"]:
                                        st.session_state["deviations"].add(log_move)

                # ── Compute visual event counts ─────────────────────────────────
                _vis_votes = 0
                _vis_motion = 0
                if "source" in current_events.columns:
                    _src = current_events["source"]
                    _vis_votes = int((_src == "Video").sum() + (_src == "Fused (Audio+Video)").sum())
                    _vis_motion = int((_src == "Audio+Motion").sum())
                    # Also count standalone Video motion events
                    if "activity_name" in current_events.columns:
                        _vis_motion += int(
                            ((current_events["activity_name"] == "Speaker Activity") & (_src == "Video")).sum()
                        )

                # ── Update Metrics ─────────────────────────────────────────────
                render_metrics(
                    fitness_raw_pct=fitness_score * 100,
                    fitness_dedup_pct=fitness_dedup * 100,
                    events_count=len(current_events),
                    agenda_count=len(st.session_state.get("agenda_activities", [])),
                    matched_count=matched_count,
                    shadow_count=shadow_count,
                    visual_votes=_vis_votes,
                    visual_motion=_vis_motion,
                )

                # ── Re-render reference BPMN with overlay if alignments changed ─
                if show_overlay and alignments:
                    render_reference_bpmn()

                # ── Warning banner ─────────────────────────────────────────────
                if fitness_dedup * 100 < 50 and not current_events.empty:
                    st.warning("Meeting deviating significantly from the planned agenda.")

                # ── Status Bar ─────────────────────────────────────────────────
                last_action = "Waiting..."
                if not mapped_events.empty:
                    last_row = mapped_events.iloc[-1]
                    act_name = last_row.get("mapped_activity", last_row["activity_name"])
                    source = last_row.get("source", "")
                    last_action = f"{act_name} ({source})"

                dot_cls = "live" if not current_events.empty else "idle"
                st.markdown(
                    f"""
<div class="status-bar">
    <span><span class="dot {dot_cls}"></span> <strong>{_sec_to_ts(current_video_time)}</strong></span>
    <span>Latest: <strong>{last_action}</strong></span>
    <span>Events: <strong>{len(current_events)}</strong></span>
</div>
""",
                    unsafe_allow_html=True,
                )

                # ── Discovered BPMN ───────────────────────────────────────────
                if not mapped_events.empty:
                    log_data = convert_to_event_log(mapped_events)
                    if log_data is not None:
                        graph, evidence_map = generate_discovered_bpmn(log_data)
                        if graph:
                            disc_bpmn_placeholder.graphviz_chart(graph, width='stretch')

                            if evidence_map:
                                with st.expander("Evidence: Why was each process added?", expanded=False):
                                    for act_name, evidence in evidence_map.items():
                                        st.markdown(
                                            f'<div class="evidence-item"><strong>{act_name}</strong><br/>{evidence}</div>',
                                            unsafe_allow_html=True,
                                        )
                        else:
                            disc_bpmn_placeholder.info("Not enough data to discover process structure yet.")
                else:
                    disc_bpmn_placeholder.info("Waiting for first event...")
    else:
        disc_bpmn_placeholder.markdown(
            """
<div class="placeholder-card">
    <h3>No Video Uploaded</h3>
    <p>Upload a meeting video (.mp4) in the sidebar to start the real-time simulation.</p>
</div>
""",
            unsafe_allow_html=True,
        )
