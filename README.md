# Meeting Process Twin

A Streamlit-based Digital Twin application for meeting process compliance. This tool compares a planned meeting agenda against the actual events discovered from a video recording of the meeting. It uses AI to generate reference models, extracts events from video/audio, and performs conformance checking to measure how well the meeting adhered to the plan.

## Features

1.  **Reference Model Generation**:
    -   Converts a text-based meeting agenda into a structured BPMN (Business Process Model and Notation) diagram.
    -   Uses OpenAI (GPT-3.5) to parse unstructured text into sequential activities.

2.  **Video & Audio Processing (Vid2Log Fusion)**:
    -   **Audio**: Extracts audio from uploaded videos and uses OpenAI Whisper to transcribe discussions.
    -   **NLP**: Uses **spaCy** to extract Actor-Verb-Object triples (e.g., "Propose Motion", "Call for Vote").
    -   **Visuals**: Uses MediaPipe Pose estimation to detect specific gestures (e.g., raising hands).
    -   **Fusion**: Merges audio and visual signals (e.g., confirming a "Vote" only if audio mentions it and hands are raised).

3.  **Real-time Simulation**:
    -   Replays the meeting events in a simulated timeline.
    -   Visualizes the process discovery incrementally as the meeting progresses.

4.  **Compliance Checking & Governance**:
    -   **Shadow Workflow Visualization**: Overlays actual execution on the planned BPMN (Green = Matched, Grey = Skipped).
    -   **Deviation Detection**: Identifies "Shadow Activities" (actions performed but not in the agenda).
    -   **Governance Interface**: Allows users to "Accept" informal deviations as valid variants directly in the sidebar.
    -   **Fitness Calculation**: Computes a fitness score (0-100%) using PM4Py's token-based replay.

## Prerequisites

-   Python 3.8+ (Recommended: 3.10 or 3.11)
-   OpenAI API Key (for GPT-3.5 and Whisper)

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
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
    -   **Toggle "Show Shadow Workflow Overlay"** to see color-coded compliance.
    -   Review detected **deviations** in the sidebar and click "✅" to accept them.

## Project Structure

-   `app.py`: Main Streamlit application entry point and UI logic.
-   `bpmn_gen.py`: Handles generation of BPMN models and **colored compliance visualization**.
-   `video_processor.py`: Manages video processing with **Vid2Log Fusion** logic.
-   `compliance_engine.py`: Contains logic for **alignment-based** compliance checking.
-   `requirements.txt`: Python package dependencies.
