import streamlit as st
import tempfile
import os
import pandas as pd
import time
from bpmn_gen import generate_agenda_bpmn, convert_to_event_log, generate_discovered_bpmn, generate_colored_bpmn
from video_processor import VideoProcessor
from compliance_engine import ComplianceEngine

# Page Config
st.set_page_config(page_title="Meeting Process Twin", layout="wide")

# Sidebar
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
uploaded_video = st.sidebar.file_uploader("Meeting Video", type=["mp4"])
agenda_text = st.sidebar.text_area("Meeting Agenda", height=200, help="Paste your meeting agenda here.")

# Governance: Detected Deviations
st.sidebar.markdown("---")
st.sidebar.subheader("Governance: Detected Deviations")

if 'deviations' not in st.session_state:
    st.session_state['deviations'] = set()
if 'accepted_deviations' not in st.session_state:
    st.session_state['accepted_deviations'] = set()

# Initialize Compliance Engine
@st.cache_resource
def load_compliance_engine():
    return ComplianceEngine()

compliance_engine = load_compliance_engine()

# Main Area
st.title("Meeting Process Twin")

# Metric Container at the top
metric_container = st.empty()

col1, col2 = st.columns(2)

# Global variables to store reference data for comparison
# Using session state to persist between reruns
if 'agenda_activities' not in st.session_state:
    st.session_state['agenda_activities'] = []
if 'reference_bpmn' not in st.session_state:
    st.session_state['reference_bpmn'] = None
if 'last_alignments' not in st.session_state:
    st.session_state['last_alignments'] = None

# Compliance Overlay Mode
show_overlay = st.toggle("Show Shadow Workflow Overlay (RQ2)", value=False)

# Left Column: Reference Model
with col1:
    st.header("Reference: Agenda BPMN")
    
    if agenda_text:
        if not api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar.")
        else:
            with st.spinner("Generating BPMN from Agenda..."):
                try:
                        # Check if we have a generated BPMN already
                        if st.session_state['reference_bpmn'] is None:
                            bpmn_viz, activities, bpmn_obj = generate_agenda_bpmn(agenda_text, api_key)
                            
                            if bpmn_viz:
                                st.session_state['agenda_activities'] = activities
                                st.session_state['reference_bpmn'] = bpmn_obj
                            else:
                                st.error("Failed to generate BPMN graph.")
                        
                        if st.session_state['reference_bpmn']:
                             # Render logic handles Overlay vs Standard below
                             pass
                         
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.info("Enter a meeting agenda in the sidebar to generate the reference model.")

# Right Column: Discovered Model (Simulation)
with col2:
    st.header("Discovered: Real-time Video BPMN")
    
    if uploaded_video is not None:
        st.success(f"Video uploaded: {uploaded_video.name}")
        
        # Add a video player
        st.video(uploaded_video)
        
        if not api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar to process the video.")
        else:
            # --- PHASE 2: PROCESSING (Once per upload) ---
            if 'video_events' not in st.session_state:
                st.info("Processing video to extract events... This may take a moment.")
                
                with st.spinner("Analyzing audio and visuals (Vid2Log Fusion)..."):
                    try:
                        # Save uploaded file to a temporary file
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        tfile.write(uploaded_video.read())
                        video_path = tfile.name
                        tfile.close()
                        
                        # Process
                        processor = VideoProcessor(api_key)
                        df = processor.process_video(video_path)
                        
                        # Store in session state
                        st.session_state['video_events'] = df
                        
                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                    finally:
                        # Cleanup temp file
                        if 'video_path' in locals() and os.path.exists(video_path):
                            os.unlink(video_path)
            
            # --- PHASE 3 & 4: SIMULATION & COMPLIANCE ---
            if 'video_events' in st.session_state:
                df_events = st.session_state['video_events']
                
                # Time Selection Slider
                def time_str_to_seconds(t_str):
                    parts = t_str.split(':')
                    if len(parts) == 2:
                        m, s = map(int, parts)
                        return m * 60 + s
                    elif len(parts) == 3:
                        h, m, s = map(int, parts)
                        return h * 3600 + m * 60 + s
                    return 0
                
                def seconds_to_time_str(seconds):
                    """Convert seconds to HH:MM:SS format"""
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
                        
                max_seconds = 0
                if not df_events.empty:
                    max_seconds = time_str_to_seconds(df_events.iloc[-1]['timestamp'])
                
                # Create time options for the slider (every 10 seconds)
                time_options = [seconds_to_time_str(s) for s in range(0, max_seconds + 60, 10)]
                if not time_options:
                    time_options = ["00:00:00"]
                
                # Interactive slider for video time with HH:MM:SS format
                selected_time_str = st.select_slider(
                    "Jump to Time (HH:MM:SS)", 
                    options=time_options, 
                    value=time_options[0],
                    key="video_time_slider"
                )
                
                current_video_time = time_str_to_seconds(selected_time_str)
                
                # Removed the button and loop for manual control via slider mostly
                # But to keep "Start Real-time Analysis" as an auto-play feature we can keep it
                # or just rely on the slider.
                # The user asked: "while video is playing... i want to be able to change time"
                # Streamlit's st.video doesn't sync bidirectional with Python backend easily in standard mode.
                # We can simulate the process state based on the slider value.
                
                # UI Containers for dynamic updates
                status_container = st.empty()
                bpmn_container = st.empty()
                
                # Simulation Logic based on Slider
                time_display = seconds_to_time_str(current_video_time)
                
                # 2. Filter Events up to selected time
                current_events = df_events[
                    df_events['timestamp'].apply(time_str_to_seconds) <= current_video_time
                ]
                
                # --- PHASE 4: COMPLIANCE CHECK ---
                fitness_score = 0.0
                alignments = []
                mapped_events = current_events
                
                if not current_events.empty and st.session_state['reference_bpmn'] and st.session_state['agenda_activities']:
                    # Map events
                    mapped_events = compliance_engine.map_events_to_agenda(
                        current_events, 
                        st.session_state['agenda_activities']
                    )
                    
                    # Convert to log for fitness calculation
                    log_data_for_fitness = convert_to_event_log(mapped_events)
                    
                    # Calculate Fitness & Alignments
                    if log_data_for_fitness is not None:
                        compliance_result = compliance_engine.calculate_fitness(
                            st.session_state['reference_bpmn'], 
                            log_data_for_fitness
                        )
                        fitness_score = compliance_result.get('score', 0.0)
                        alignments = compliance_result.get('alignments', [])
                        st.session_state['last_alignments'] = alignments
                        
                        # Analyze Alignments for Governance
                        for align in alignments:
                            for log_move, model_move in align['alignment']:
                                if (model_move is None or model_move == '>>') and log_move is not None:
                                    # Shadow activity (Log Move Only)
                                    if log_move not in st.session_state['accepted_deviations']:
                                        st.session_state['deviations'].add(log_move)
                
                # Update Metric Card
                fitness_percent = fitness_score * 100
                delta_color = "normal"
                warning_msg = ""
                
                if fitness_percent < 50 and not current_events.empty:
                    delta_color = "inverse"
                    warning_msg = "⚠️ Meeting deviating from Agenda!"
                    
                metric_container.metric(
                    label="Process Fitness Score (Behavioral Alignment)",
                    value=f"{fitness_percent:.1f}%",
                    delta=f"{fitness_percent-100:.1f}%" if fitness_percent < 100 else "Perfect",
                    delta_color=delta_color
                )
                if warning_msg:
                    st.warning(warning_msg)

                # Update Status Display
                last_action = "Waiting..."
                if not mapped_events.empty:
                    last_row = mapped_events.iloc[-1]
                    act_name = last_row.get('mapped_activity', last_row['activity_name'])
                    original = last_row['activity_name']
                    source = last_row['source']
                    last_action = f"{act_name} (Source: {source})"
                    if 'mapped_activity' in last_row and last_row['mapped_activity'] != last_row['activity_name']:
                        last_action += f" [Mapped from: {original}]"
                
                status_container.markdown(
                    f"""
                    <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 10px;">
                        <strong>Video Time:</strong> {time_display} <br>
                        <strong>Latest Event:</strong> {last_action} <br>
                        <strong>Events Detected:</strong> {len(current_events)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # 3. Discover & Update BPMN
                # If Overlay Mode: Update Left Column with Colored Graph
                if show_overlay and st.session_state['reference_bpmn'] and alignments:
                     colored_viz, _ = generate_colored_bpmn(st.session_state['reference_bpmn'], alignments)
                     # The bottom logic for Col1 will pick up 'last_alignments' on rerun
                     pass 

                if not mapped_events.empty:
                    log_data = convert_to_event_log(mapped_events)
                    if log_data is not None:
                        graph, evidence_map = generate_discovered_bpmn(log_data)
                        if graph:
                            bpmn_container.graphviz_chart(graph, width='stretch')
                            
                            # Display Evidence Panel - Click to see why each process was added
                            if evidence_map:
                                with st.expander("📋 **Click to see Evidence: Why was each process added?**", expanded=False):
                                    st.markdown("**Select an activity to see the justification:**")
                                    for activity_name, evidence in evidence_map.items():
                                        st.markdown(f"**🔹 {activity_name}**")
                                        st.info(f"_{evidence}_")
                                        st.markdown("---")
                        else:
                            bpmn_container.info("Not enough data to discover process structure yet.")
                else:
                    bpmn_container.info("Waiting for first event...")

    else:
        st.markdown(
            """
            <div style="border: 2px dashed #ccc; padding: 20px; text-align: center; height: 300px; display: flex; align-items: center; justify-content: center; border-radius: 10px; margin-bottom: 20px;">
                <div>
                    <h3>Video Player Placeholder</h3>
                    <p>Upload a video to start the real-time simulation.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Render Left Column content (Ref BPMN or Overlay)
# We do this at the end or use placeholders. 
# Best way: Use a placeholder defined early in Col1.

with col1:
    # Clear previous content if any (Streamlit runs top to bottom, but we are back in col1 context?)
    # No, 'with col1:' just appends.
    # To properly update, we should have used a placeholder.
    # Let's assume the user toggles and it re-runs.
    
    if st.session_state.get('reference_bpmn'):
        alignments = st.session_state.get('last_alignments')
        
        if show_overlay and alignments:
             st.subheader("Shadow Workflow Overlay")
             colored_viz, compliance_info = generate_colored_bpmn(st.session_state['reference_bpmn'], alignments)
             st.graphviz_chart(colored_viz, width='stretch')
             st.caption("🟢 Green: Executed | ⬜ Grey: Skipped | 🔴 Red (Sidebar): Deviations")
             
             # Display Compliance Status Panel
             if compliance_info:
                 with st.expander("📋 **Click to see Compliance Status for each Agenda Item**", expanded=False):
                     for activity_name, status in compliance_info.items():
                         if status == "executed":
                             st.success(f"✅ **{activity_name}**: Matched with video events")
                         elif status == "skipped":
                             st.warning(f"⚠️ **{activity_name}**: Planned but not detected in video")
                         else:
                             st.info(f"ℹ️ **{activity_name}**: {status}")
             
        elif show_overlay and not alignments:
             st.info("Please run the simulation first to generate compliance data for the overlay.")
             # Fallback to standard view so it's not empty
             from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
             parameters = {bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg"}
             gviz = bpmn_visualizer.apply(st.session_state['reference_bpmn'], parameters=parameters)
             gviz.attr(rankdir='TB')
             st.graphviz_chart(gviz, width='stretch')
             
        else:
             # Standard view
             from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
             parameters = {bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg"}
             gviz = bpmn_visualizer.apply(st.session_state['reference_bpmn'], parameters=parameters)
             gviz.attr(rankdir='TB')
             st.graphviz_chart(gviz, width='stretch')


# Governance Sidebar Logic (RQ3)
if st.session_state['deviations']:
    st.sidebar.warning(f"Detected {len(st.session_state['deviations'])} Shadow Activities")
    
    # Iterate over a copy to modify set during iteration
    for dev in list(st.session_state['deviations']):
        col_dev1, col_dev2 = st.sidebar.columns([3, 1])
        col_dev1.write(f"**{dev}**")
        
        if col_dev2.button("✅", key=f"accept_{dev}", help="Accept as formal variant"):
            st.session_state['accepted_deviations'].add(dev)
            st.session_state['deviations'].remove(dev)
            st.rerun()
            
if st.session_state['accepted_deviations']:
    st.sidebar.success("Accepted Variants:")
    for dev in st.session_state['accepted_deviations']:
        st.sidebar.write(f"- {dev}")
