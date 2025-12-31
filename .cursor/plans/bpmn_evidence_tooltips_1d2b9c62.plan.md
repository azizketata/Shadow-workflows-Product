---
name: BPMN Evidence Tooltips
overview: Implement interactive tooltips for BPMN nodes that display the "evidence" (audio transcript or visual cue) justifying why the process was added.
todos:
  - id: tooltips-discovered
    content: Update generate_discovered_bpmn to inject tooltips from event log details
    status: completed
  - id: tooltips-reference
    content: Update generate_colored_bpmn to inject compliance status tooltips
    status: completed
    dependencies:
      - tooltips-discovered
---

# BPMN Evidence Tooltips Implementation Plan

To address the user story *"the user should be able to see... a certain reason or a pop up that justifies why that process was added"*, we will implement interactive tooltips on the BPMN visualization. Since Streamlit's native graph support is static, we will use SVG tooltips which natively render as "popups" when hovering over nodes.

## 1. Data Processing

-   **Goal**: Ensure the "evidence" (text transcript or visual detection details) is available during BPMN generation.
-   **Files**: `bpmn_gen.py`
-   **Details**:
    -   The `convert_to_event_log` function already preserves the `details` column from the input DataFrame. We will utilize this.

## 2. Discovered BPMN Visualization (Real-time)

-   **Goal**: Inject evidence into the Discovered BPMN nodes.
-   **Files**: `bpmn_gen.py` (`generate_discovered_bpmn`)
-   **Implementation**:
    -   Create a mapping of `Activity Name -> Evidence Details` from the `log_df`.
    -   Iterate through the generated Graphviz `body`.
    -   For each node (Activity), inject a `tooltip="Evidence: ..."` attribute containing the source text (e.g., "Audio: 'I move to vote...'") or visual cue (e.g., "Visual: Hand Raised").

## 3. Reference BPMN Visualization (Overlay)

-   **Goal**: Explain the status of nodes in the Reference model.
-   **Files**: `bpmn_gen.py` (`generate_colored_bpmn`)
-   **Implementation**:
    -   Enhance the existing coloring logic.
    -   For **Green (Matched)** nodes: Add tooltip showing the matching event's evidence from the video.
    -   For **Grey (Skipped)** nodes: Add tooltip "Planned in Agenda but not detected in Video".
    -   For **Red (Deviations)** (if visualized): Add tooltip with the deviation reason.

## 4. UI Update

-   **Goal**: Ensure the graph renders as SVG to support tooltips.
-   **Files**: `app.py`
-   **Implementation**: Verify `st.graphviz_chart` or `st.image` is used correctly to support SVG interactivity (Graphviz usually handles this automatically in browsers).

## Verification