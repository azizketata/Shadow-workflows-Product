import streamlit as st
import tempfile
import os
import pandas as pd
from bpmn_gen import generate_agenda_bpmn
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

# Right Column: Discovered Model (Placeholder -> Implemented)
with col2:
    st.header("Discovered: Real-time Video BPMN")
    
    # Placeholder for Video Player (Visual Only)
    st.markdown(
        """
        <div style="border: 2px dashed #ccc; padding: 20px; text-align: center; height: 300px; display: flex; align-items: center; justify-content: center; border-radius: 10px; margin-bottom: 20px;">
            <div>
                <h3>Video Player Placeholder</h3>
                <p>Real-time playback will be implemented in the next phase.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if uploaded_video is not None:
        st.success(f"Video uploaded: {uploaded_video.name}")
        
        if not api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar to process the video.")
        else:
            # Check if we already processed this video or if user wants to reprocess
            # We use a simple key in session state for now. 
            # Ideally we'd hash the file, but for now we rely on the object existence
            
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
            
            # Display Results
            if 'video_events' in st.session_state:
                st.subheader("Extracted Events Data")
                st.dataframe(st.session_state['video_events'], use_container_width=True)
                
                # Option to clear/reprocess
                if st.button("Clear Processed Data"):
                    del st.session_state['video_events']
                    st.rerun()
