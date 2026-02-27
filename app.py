"""Meeting Process Twin — Slim Orchestrator.

Wires together UI components, pipeline phases, and feature modules.
All heavy logic lives in pipeline/, ui/, and features/ packages.
"""

import streamlit as st
import time

from ui.styles import inject_styles, HERO_HTML
from ui.sidebar import render_sidebar, render_sidebar_dynamic
from ui.metrics import render_metrics_html, compute_visual_counts
from ui.bpmn_views import render_reference_bpmn, render_discovered_bpmn
from pipeline.state import (
    init_state,
    get_video_events, set_video_events,
    get_abstracted_events, set_abstracted_events,
    get_mapped_events, set_mapped_events,
    get_reference_bpmn, set_reference_bpmn,
    get_agenda_activities, set_agenda_activities,
    get_alignments, set_alignments,
    get_abstraction_cache, get_accepted_deviations,
    add_deviation, append_debug_log,
)
from pipeline.time_utils import ts_to_seconds, seconds_to_ts, make_time_options
from pipeline.orchestrator import (
    run_video_processing, run_abstraction, run_compliance, detect_deviations,
)
from bpmn_gen import generate_agenda_bpmn, generate_colored_bpmn
from compliance_engine import ComplianceEngine

# ── Feature modules ───────────────────────────────────────────────────────────
from features.chapters import build_chapters, render_chapters
from features.summary import get_or_generate_summary, render_summary
from features.report_card import generate_report_card, render_report_card
from features.shadow_timeline import build_timeline_data, render_timeline
from features.transcript import build_annotated_transcript, render_transcript
from features.voting import extract_voting_records, render_voting_records
from features.duration import compute_durations, render_duration_analysis
from features.deep_links import apply_deep_link_params
from features.trends import render_trend_dashboard
from features.alerts import render_alerts

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Process Twin",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles + State ────────────────────────────────────────────────────────────
inject_styles()
init_state()

# ── Sidebar (config widgets) ─────────────────────────────────────────────────
config = render_sidebar()

# ── Debug logger (closure over config) ────────────────────────────────────────
def log_debug(message: str):
    if config["debug_enabled"]:
        ts = time.strftime("%H:%M:%S")
        append_debug_log(f"[{ts}] {message}")

# ── Compliance engine (cached per SBERT model) ───────────────────────────────
@st.cache_resource
def load_compliance_engine(model_name):
    return ComplianceEngine(sbert_model_name=model_name)

compliance_engine = load_compliance_engine(config["sbert_model_choice"])

# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER + METRICS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(HERO_HTML, unsafe_allow_html=True)
metric_placeholder = st.empty()
render_metrics_html(metric_placeholder)  # initial empty render

show_overlay = st.toggle("Show Shadow Workflow Overlay", value=False)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">📋 Reference: Agenda BPMN</div>', unsafe_allow_html=True)
    ref_bpmn_placeholder = st.empty()

with col2:
    st.markdown('<div class="section-header">🔍 Discovered: Real-time Video BPMN</div>', unsafe_allow_html=True)
    disc_bpmn_placeholder = st.empty()

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — AGENDA BPMN (Left Column)
# ══════════════════════════════════════════════════════════════════════════════
if config["agenda_text"]:
    if not config["api_key"]:
        with col1:
            ref_bpmn_placeholder.warning("Enter your OpenAI API Key in the sidebar to generate the agenda BPMN.")
    else:
        if get_reference_bpmn() is None:
            with col1:
                with st.spinner("Generating BPMN from agenda..."):
                    try:
                        bpmn_viz, activities, bpmn_obj = generate_agenda_bpmn(
                            config["agenda_text"], config["api_key"],
                        )
                        if bpmn_viz and bpmn_obj:
                            set_agenda_activities(activities)
                            set_reference_bpmn(bpmn_obj)
                            log_debug(f"Agenda BPMN generated — {len(activities)} activities")
                        else:
                            ref_bpmn_placeholder.error("Failed to generate BPMN from agenda.")
                            log_debug("Agenda BPMN generation returned None.")
                    except Exception as e:
                        ref_bpmn_placeholder.error(f"Error: {e}")
else:
    with col1:
        ref_bpmn_placeholder.markdown(
            '<div class="placeholder-card"><h3>No Agenda</h3>'
            "<p>Paste your meeting agenda in the sidebar.</p></div>",
            unsafe_allow_html=True,
        )

# Render reference BPMN (standard or overlay)
with col1:
    render_reference_bpmn(ref_bpmn_placeholder, show_overlay)

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2-4 — VIDEO PIPELINE (Right Column)
# ══════════════════════════════════════════════════════════════════════════════
with col2:
    uploaded_video = config["uploaded_video"]

    if uploaded_video is not None:
        st.video(uploaded_video)

        if not config["api_key"] and not config["use_local_whisper"]:
            disc_bpmn_placeholder.warning(
                "Enter your OpenAI API Key or enable local Whisper in the sidebar."
            )
        else:
            # ── Phase 2: Process video (once) ─────────────────────────────────
            if get_video_events() is None:
                with st.spinner("Analysing audio and visual streams..."):
                    try:
                        progress_bar = st.progress(0, text="Starting video analysis...")
                        def _st_progress(percent, text=""):
                            progress_bar.progress(min(percent, 100), text=text)

                        df = run_video_processing(
                            video_file=uploaded_video,
                            api_key=config["api_key"],
                            use_local_whisper=config["use_local_whisper"],
                            local_whisper_model=config["local_whisper_model"],
                            debug=config["debug_enabled"],
                            progress_callback=_st_progress,
                            log_fn=log_debug,
                        )
                        progress_bar.empty()
                        set_video_events(df)
                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                        log_debug(f"Video processing error: {e}")

            # ── Simulation Slider ─────────────────────────────────────────────
            if get_video_events() is not None:
                df_events = get_video_events()

                max_seconds = 0
                if not df_events.empty:
                    max_seconds = ts_to_seconds(df_events.iloc[-1]["timestamp"])

                time_options = make_time_options(max_seconds)
                selected_time_str = st.select_slider(
                    "Jump to time", options=time_options,
                    value=time_options[-1], key="video_time_slider",
                )
                current_video_time = ts_to_seconds(selected_time_str)

                # Filter events up to selected time
                current_events = df_events[
                    df_events["timestamp"].apply(ts_to_seconds) <= current_video_time
                ]
                log_debug(f"Slider: {selected_time_str} — {len(current_events)} events in scope")

                # ── Phase 3: Semantic Abstraction ─────────────────────────────
                abstracted_events = run_abstraction(
                    events_df=current_events,
                    agenda_activities=get_agenda_activities(),
                    config=config,
                    cache=get_abstraction_cache(),
                    compliance_engine=compliance_engine,
                    log_fn=log_debug,
                )
                set_abstracted_events(abstracted_events)

                # ── Phase 4: Compliance Check ─────────────────────────────────
                results = run_compliance(
                    abstracted_df=abstracted_events,
                    reference_bpmn=get_reference_bpmn(),
                    agenda_activities=get_agenda_activities(),
                    sbert_threshold=config["sbert_threshold"],
                    compliance_engine=compliance_engine,
                    log_fn=log_debug,
                )

                mapped_events = results["mapped_events"]
                set_mapped_events(mapped_events)
                set_alignments(results["alignments"])
                st.session_state["_last_fitness_dedup"] = results["fitness_dedup"]

                # Detect shadow activities for governance
                new_devs = detect_deviations(results["alignments"], get_accepted_deviations())
                for dev in new_devs:
                    add_deviation(dev)

                # ── Visual event counts ───────────────────────────────────────
                vis_votes, vis_motion = compute_visual_counts(current_events)

                # ── Update Metrics ────────────────────────────────────────────
                render_metrics_html(
                    metric_placeholder,
                    fitness_raw_pct=results["fitness_raw"] * 100,
                    fitness_dedup_pct=results["fitness_dedup"] * 100,
                    events_count=len(current_events),
                    agenda_count=len(get_agenda_activities()),
                    matched_count=results["matched_count"],
                    shadow_count=results["shadow_count"],
                    visual_votes=vis_votes,
                    visual_motion=vis_motion,
                )

                # ── Re-render reference BPMN overlay ──────────────────────────
                if show_overlay and results["alignments"]:
                    with col1:
                        render_reference_bpmn(ref_bpmn_placeholder, show_overlay)

                # ── Warning banner ────────────────────────────────────────────
                if results["fitness_dedup"] * 100 < 50 and not current_events.empty:
                    st.warning("Meeting deviating significantly from the planned agenda.")

                # ── Status Bar ────────────────────────────────────────────────
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
    <span><span class="dot {dot_cls}"></span> <strong>{seconds_to_ts(current_video_time)}</strong></span>
    <span>Latest: <strong>{last_action}</strong></span>
    <span>Events: <strong>{len(current_events)}</strong></span>
</div>
""",
                    unsafe_allow_html=True,
                )

                # ── Discovered BPMN ──────────────────────────────────────────
                render_discovered_bpmn(disc_bpmn_placeholder, mapped_events)
    else:
        disc_bpmn_placeholder.markdown(
            '<div class="placeholder-card"><h3>No Video Uploaded</h3>'
            "<p>Upload a meeting video (.mp4) in the sidebar to start the real-time simulation.</p></div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
#  CITIZEN FEATURES (tabbed interface below main content)
# ══════════════════════════════════════════════════════════════════════════════
mapped = get_mapped_events()
has_data = mapped is not None and hasattr(mapped, "empty") and not mapped.empty

st.markdown("---")
tabs = st.tabs([
    "Chapters", "Summary", "Report Card", "Shadow Timeline",
    "Transcript", "Voting", "Duration", "Trends", "Alerts",
])

with tabs[0]:  # Chapters
    if has_data:
        chapters = build_chapters(mapped, get_agenda_activities())
        render_chapters(chapters, config["uploaded_video"])
    else:
        st.info("Process a video to view meeting chapters.")

with tabs[1]:  # Summary
    if has_data and config["api_key"]:
        summary = get_or_generate_summary(
            mapped, get_agenda_activities(), config["api_key"],
            model=config["openai_abstraction_model"],
        )
        render_summary(summary)
    elif has_data:
        st.info("Provide an OpenAI API key to generate the meeting summary.")
    else:
        st.info("Process a video to generate a meeting summary.")

with tabs[2]:  # Report Card
    if has_data:
        alignments = get_alignments()
        compliance_info = {}
        if alignments:
            _, compliance_info = generate_colored_bpmn(
                get_reference_bpmn(), alignments,
                accepted_deviations=get_accepted_deviations(),
            )
            compliance_info = compliance_info or {}
        # Get fitness_dedup from session state (stored during pipeline run)
        _fitness_dedup = st.session_state.get("_last_fitness_dedup", 0.0)
        card = generate_report_card(
            fitness_dedup=_fitness_dedup,
            compliance_info=compliance_info,
            alignments=alignments or [],
            mapped_events=mapped,
            agenda_activities=get_agenda_activities(),
        )
        render_report_card(card)
    else:
        st.info("Process a video to view the compliance report card.")

with tabs[3]:  # Shadow Timeline
    if has_data:
        timeline_data = build_timeline_data(mapped, get_agenda_activities())
        render_timeline(timeline_data)
    else:
        st.info("Process a video to view the shadow workflow timeline.")

with tabs[4]:  # Transcript
    if has_data:
        transcript = build_annotated_transcript(mapped)
        render_transcript(transcript)
    else:
        st.info("Process a video to view the annotated transcript.")

with tabs[5]:  # Voting
    if has_data:
        records = extract_voting_records(mapped)
        render_voting_records(records)
    else:
        st.info("Process a video to view voting records.")

with tabs[6]:  # Duration
    if has_data:
        durations = compute_durations(mapped, get_agenda_activities())
        render_duration_analysis(durations)
    else:
        st.info("Process a video to view duration analysis.")

with tabs[7]:  # Trends
    render_trend_dashboard()

with tabs[8]:  # Alerts
    render_alerts(mapped_events=mapped if has_data else None)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — DYNAMIC SECTIONS (after main processing)
# ══════════════════════════════════════════════════════════════════════════════
render_sidebar_dynamic()
