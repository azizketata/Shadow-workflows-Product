# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2023-10-27

### Added
-   **Core Application**: Initial release of the Meeting Process Twin Streamlit app.
-   **Agenda to BPMN**: Functionality to convert text agendas into BPMN diagrams using OpenAI GPT-3.5 (`bpmn_gen.py`).
-   **Video Processing**:
    -   Audio extraction and transcription using OpenAI Whisper (`video_processor.py`).
    -   Gesture detection (Hand Raising) using MediaPipe Pose for "Voting" events.
-   **Compliance Engine**:
    -   Semantic mapping of discovered events to agenda items using `sentence-transformers`.
    -   Conformance checking and fitness score calculation using `pm4py`.
-   **Simulation**: Real-time playback feature to visualize the meeting progress and updating process model.

