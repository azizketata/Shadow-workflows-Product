"""BPMN rendering helpers for reference and discovered process models."""

import streamlit as st
from bpmn_gen import generate_colored_bpmn, generate_discovered_bpmn, convert_to_event_log
from pipeline.state import get_reference_bpmn, get_alignments, get_accepted_deviations


def render_reference_bpmn(placeholder, show_overlay: bool):
    """Render the reference BPMN into *placeholder*, either colored (overlay) or plain."""
    bpmn_obj = get_reference_bpmn()
    if not bpmn_obj:
        return

    alignments = get_alignments()

    if show_overlay and alignments:
        colored_viz, compliance_info = generate_colored_bpmn(
            bpmn_obj, alignments,
            accepted_deviations=get_accepted_deviations(),
        )
        if colored_viz:
            placeholder.graphviz_chart(colored_viz, width="stretch")
            st.caption("🟢 Executed  |  ⬜ Skipped  |  🔶 Shadow (see sidebar)")
            if compliance_info:
                with st.expander("Compliance Status per Agenda Item", expanded=False):
                    for act, status in compliance_info.items():
                        if status == "executed":
                            st.success(f"**{act}** — matched")
                        elif status == "skipped":
                            st.warning(f"**{act}** — not detected")
                        elif status == "accepted":
                            st.error(f"**{act}** — accepted shadow (added to model)")
                        elif status == "deviation":
                            st.error(f"**{act}** — shadow workflow (deviation)")
                        else:
                            st.info(f"**{act}** — {status}")
        else:
            placeholder.info("Overlay could not be generated.")

    elif show_overlay and not alignments:
        from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
        params = {bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg"}
        gviz = bpmn_visualizer.apply(bpmn_obj, parameters=params)
        gviz.attr(rankdir="TB")
        placeholder.graphviz_chart(gviz, width="stretch")
        st.info("Process the video first to generate overlay data.")
    else:
        from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
        params = {bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg"}
        gviz = bpmn_visualizer.apply(bpmn_obj, parameters=params)
        gviz.attr(rankdir="TB")
        placeholder.graphviz_chart(gviz, width="stretch")


def render_discovered_bpmn(placeholder, mapped_events):
    """Render discovered BPMN from mapped events into *placeholder*."""
    if mapped_events is None or mapped_events.empty:
        placeholder.info("Waiting for first event...")
        return

    log_data = convert_to_event_log(mapped_events)
    if log_data is None:
        placeholder.info("Not enough data to discover process structure yet.")
        return

    graph, evidence_map = generate_discovered_bpmn(log_data)
    if graph:
        placeholder.graphviz_chart(graph, width="stretch")
        if evidence_map:
            with st.expander("Evidence: Why was each process added?", expanded=False):
                for act_name, evidence in evidence_map.items():
                    st.markdown(
                        f'<div class="evidence-item"><strong>{act_name}</strong><br/>{evidence}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        placeholder.info("Not enough data to discover process structure yet.")
