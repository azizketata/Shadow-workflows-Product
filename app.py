import streamlit as st
import tempfile
import os
from bpmn_gen import generate_agenda_bpmn

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

# Right Column: Discovered Model (Placeholder)
with col2:
    st.header("Discovered: Real-time Video BPMN")
    
    # Placeholder for Video Player
    st.markdown(
        """
        <div style="border: 2px dashed #ccc; padding: 20px; text-align: center; height: 400px; display: flex; align-items: center; justify-content: center; border-radius: 10px;">
            <div>
                <h3>Video Player Placeholder</h3>
                <p>Video processing and discovery model will appear here.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if uploaded_video is not None:
        st.success(f"Video uploaded: {uploaded_video.name}")
        # In the future, we would process the video here.

