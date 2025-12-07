import streamlit as st
import tempfile
import os
import pandas as pd
import time
from bpmn_gen import generate_agenda_bpmn, convert_to_event_log, generate_discovered_bpmn
from video_processor import VideoProcessor

# Page Config
st.set_page_config(page_title="Meeting Process Twin", layout="wide")

# Sidebar
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
uploaded_video = st.sidebar.file_uploader("Meeting Video", type=["mp4"])
agenda_text = st.sidebar.text_area("Meeting Agenda", height=200, help="Paste your meeting agenda here.")

# Main Area
st.title("Meeting Process Twin")

col1, col2 = st.columns(2)

# Left Column: Reference Model
with col1:
    st.header("Reference: Agenda BPMN")
    
    if agenda_text:
        if not api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar.")
        else:
            with st.spinner("Generating BPMN from Agenda..."):
                try:
                    bpmn_graph = generate_agenda_bpmn(agenda_text, api_key)
                    if bpmn_graph:
                        st.graphviz_chart(bpmn_graph, use_container_width=True)
                    else:
                        st.error("Failed to generate BPMN graph.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.info("Enter a meeting agenda in the sidebar to generate the reference model.")

# Right Column: Discovered Model (Simulation)
with col2:
    st.header("Discovered: Real-time Video BPMN")
    
    if uploaded_video is not None:
        st.success(f"Video uploaded: {uploaded_video.name}")
        
        if not api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar to process the video.")
        else:
            # --- PHASE 2: PROCESSING (Once per upload) ---
            if 'video_events' not in st.session_state:
                st.info("Processing video to extract events... This may take a moment.")
                
                with st.spinner("Analyzing audio and visuals..."):
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
            
            # --- PHASE 3: SIMULATION ---
            if 'video_events' in st.session_state:
                df_events = st.session_state['video_events']
                
                # Controls
                col_ctrl1, col_ctrl2 = st.columns([1, 2])
                with col_ctrl1:
                    if st.button("Start Real-time Analysis"):
                        st.session_state['simulation_active'] = True
                
                # UI Containers for dynamic updates
                status_container = st.empty()
                bpmn_container = st.empty()
                
                # Initial View (All events if not simulating, or empty)
                if 'simulation_active' not in st.session_state:
                     st.write("Click 'Start Analysis' to simulate real-time discovery.")
                     st.dataframe(df_events, use_container_width=True, height=200)

                # Simulation Loop
                if st.session_state.get('simulation_active'):
                    # Determine duration (convert last timestamp to seconds)
                    # For simplicity, we'll iterate through the event list directly or time steps
                    # Time-step approach is better for "Real-time" feel
                    
                    # Convert timestamps to seconds for easier comparison
                    def time_str_to_seconds(t_str):
                        m, s = map(int, t_str.split(':'))
                        return m * 60 + s
                        
                    max_seconds = 0
                    if not df_events.empty:
                        max_seconds = time_str_to_seconds(df_events.iloc[-1]['timestamp'])
                    
                    # Simulation settings
                    speed_multiplier = 2  # Run 2x faster than real-time
                    step_size = 5 # Update every 5 seconds of video time
                    
                    # Loop
                    current_video_time = 0
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    while current_video_time <= max_seconds + step_size:
                        # 1. Update Status
                        mins, secs = divmod(current_video_time, 60)
                        time_display = f"{int(mins):02d}:{int(secs):02d}"
                        
                        # 2. Filter Events up to current time
                        # We need to filter based on string comparison or pre-calculated seconds
                        # Let's do a quick lambda for this filter
                        current_events = df_events[
                            df_events['timestamp'].apply(time_str_to_seconds) <= current_video_time
                        ]
                        
                        last_action = "Waiting..."
                        if not current_events.empty:
                            last_action = f"{current_events.iloc[-1]['activity_name']} ({current_events.iloc[-1]['source']})"
                        
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
                        
                        # 3. Discover & Update BPMN (Incremental)
                        if not current_events.empty:
                            # We need at least a few events to form a graph
                            log_data = convert_to_event_log(current_events)
                            if log_data is not None:
                                graph = generate_discovered_bpmn(log_data)
                                if graph:
                                    bpmn_container.graphviz_chart(graph, use_container_width=True)
                                else:
                                    bpmn_container.info("Not enough data to discover process structure yet.")
                        else:
                            bpmn_container.info("Waiting for first event...")

                        # Update Progress
                        if max_seconds > 0:
                            progress = min(current_video_time / max_seconds, 1.0)
                            progress_bar.progress(progress)

                        # Sleep
                        time.sleep(1 / speed_multiplier) # Sleep less for faster playback
                        
                        # Increment
                        current_video_time += step_size
                    
                    st.success("Simulation Complete")
                    st.session_state['simulation_active'] = False # Reset
                
                # Option to clear
                if st.button("Clear Data & Reset"):
                    del st.session_state['video_events']
                    if 'simulation_active' in st.session_state:
                        del st.session_state['simulation_active']
                    st.rerun()

    else:
        # Placeholder for Video Player (Visual Only)
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
