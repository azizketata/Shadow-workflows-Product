# Meeting Process Twin

A Streamlit-based Digital Twin application for meeting process compliance. This tool compares a planned meeting agenda against the actual events discovered from a video recording of the meeting. It uses AI to generate reference models, extracts events from video/audio, and performs conformance checking to measure how well the meeting adhered to the plan.

## Features

1.  **Reference Model Generation**:
    -   Converts a text-based meeting agenda into a structured BPMN (Business Process Model and Notation) diagram.
    -   Uses OpenAI (GPT-3.5) to parse unstructured text into sequential activities.

2.  **Video & Audio Processing**:
    -   **Audio**: Extracts audio from uploaded videos and uses OpenAI Whisper to transcribe discussions.
    -   **Visuals**: Uses MediaPipe Pose estimation to detect specific gestures (e.g., raising hands for "Voting").

3.  **Real-time Simulation**:
    -   Replays the meeting events in a simulated timeline.
    -   Visualizes the process discovery incrementally as the meeting progresses.

4.  **Compliance Checking**:
    -   **Semantic Mapping**: Maps discovered events (like general "Discussion") to specific agenda items using Semantic Search (SBERT).
    -   **Fitness Calculation**: Computes a fitness score (0-100%) using PM4Py's token-based replay to quantify alignment between the agenda and reality.
    -   **Alerts**: visually warns when the meeting deviates significantly from the agenda.

## Prerequisites

-   Python 3.8+
-   OpenAI API Key (for GPT-3.5 and Whisper)

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: You may need to install ffmpeg on your system for `moviepy` to work correctly.*

## Usage

1.  Run the Streamlit application:

```bash
streamlit run app.py
```

2.  **Sidebar Configuration**:
    -   Enter your **OpenAI API Key**.
    -   Upload a **Meeting Video** (`.mp4`).
    -   Paste the **Meeting Agenda** text.

3.  **Workflow**:
    -   The app will automatically generate the **Reference BPMN** from your agenda.
    -   Once a video is uploaded, it processes the file (this may take a moment) to extract events.
    -   Click **"Start Real-time Analysis"** to begin the simulation.
    -   Observe the **Process Fitness Score** and the **Discovered BPMN** evolving in real-time.

## Project Structure

-   `app.py`: Main Streamlit application entry point and UI logic.
-   `bpmn_gen.py`: Handles generation of BPMN models from text (Agenda) and event logs.
-   `video_processor.py`: Manages video processing (audio extraction, transcription, pose detection).
-   `compliance_engine.py`: Contains logic for semantic mapping of events and fitness calculation.
-   `requirements.txt`: Python package dependencies.

