import openai
import pm4py
from pm4py.objects.bpmn.obj import BPMN
import json
import tempfile
import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

def generate_agenda_bpmn(agenda_text, api_key):
    """
    Generates a BPMN visualization from a meeting agenda text using OpenAI and pm4py.
    
    Args:
        agenda_text (str): The text of the meeting agenda.
        api_key (str): OpenAI API key.
        
    Returns:
        tuple: (graphviz.Digraph, list of activities, BPMN object)
    """
    if not agenda_text or not api_key:
        return None, [], None

    try:
        client = openai.OpenAI(api_key=api_key)

        prompt = f"""Extract a list of activities/topics from the following meeting agenda.
Return a JSON object with this structure:
{{
  "activities": [
    {{"label": "Call to Order", "group": null}},
    {{"label": "Consent Item A", "group": "consent"}},
    {{"label": "Consent Item B", "group": "consent"}},
    {{"label": "Staff Report", "group": null}}
  ]
}}

Rules:
- Each activity has a concise label (3-6 words)
- Activities that can occur simultaneously (e.g., consent agenda sub-items, concurrent staff reports) should share the same "group" string
- Sequential activities (most items) should have "group": null
- Include sub-items if they represent distinct steps (e.g. "Open Public Hearing", "Close Public Hearing")
- Ensure the order matches the agenda exactly

Agenda:
{agenda_text}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured process activities from meeting agendas. Be precise and concise. Return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        content = response.choices[0].message.content

        # Parse response — handle both new format (dict with "activities") and legacy (plain array)
        # Try dict format first
        activities_data = None
        try:
            brace_start = content.find('{')
            brace_end = content.rfind('}') + 1
            if brace_start != -1 and brace_end > 0:
                parsed = json.loads(content[brace_start:brace_end])
                if isinstance(parsed, dict) and "activities" in parsed:
                    activities_data = parsed["activities"]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try plain array
        if activities_data is None:
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("Could not find JSON in response")
            parsed = json.loads(content[start_idx:end_idx])
            if isinstance(parsed, list):
                activities_data = [
                    {"label": a, "group": None} if isinstance(a, str) else a
                    for a in parsed
                ]
            else:
                raise ValueError("Unexpected JSON structure in response")

        activities = [a["label"] if isinstance(a, dict) else str(a) for a in activities_data]

        # Build BPMN with parallel gateways for grouped activities
        bpmn_graph = BPMN()

        start_event = BPMN.StartEvent(name="Start")
        bpmn_graph.add_node(start_event)

        end_event = BPMN.EndEvent(name="End")
        bpmn_graph.add_node(end_event)

        previous_node = start_event

        i = 0
        while i < len(activities_data):
            act = activities_data[i]
            group = act.get("group") if isinstance(act, dict) else None

            if group is None:
                # Sequential task
                label = act["label"] if isinstance(act, dict) else str(act)
                task = BPMN.Task(name=label)
                bpmn_graph.add_node(task)
                bpmn_graph.add_flow(BPMN.SequenceFlow(previous_node, task))
                previous_node = task
                i += 1
            else:
                # Collect all consecutive activities in this group
                group_items = []
                while i < len(activities_data):
                    a = activities_data[i]
                    g = a.get("group") if isinstance(a, dict) else None
                    if g != group:
                        break
                    group_items.append(a)
                    i += 1

                if len(group_items) == 1:
                    label = group_items[0]["label"] if isinstance(group_items[0], dict) else str(group_items[0])
                    task = BPMN.Task(name=label)
                    bpmn_graph.add_node(task)
                    bpmn_graph.add_flow(BPMN.SequenceFlow(previous_node, task))
                    previous_node = task
                else:
                    # Parallel gateway: fork -> parallel tasks -> join
                    fork = BPMN.ParallelGateway(name=f"Fork_{group}")
                    bpmn_graph.add_node(fork)
                    bpmn_graph.add_flow(BPMN.SequenceFlow(previous_node, fork))

                    join = BPMN.ParallelGateway(name=f"Join_{group}")
                    bpmn_graph.add_node(join)

                    for item in group_items:
                        label = item["label"] if isinstance(item, dict) else str(item)
                        task = BPMN.Task(name=label)
                        bpmn_graph.add_node(task)
                        bpmn_graph.add_flow(BPMN.SequenceFlow(fork, task))
                        bpmn_graph.add_flow(BPMN.SequenceFlow(task, join))

                    previous_node = join

        # Connect last node to end event
        final_flow = BPMN.SequenceFlow(previous_node, end_event)
        bpmn_graph.add_flow(final_flow)

        # layout top-to-bottom
        # pm4py's view_bpmn usually displays it. To get the graphviz object we can use the visualizer directly.
        from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
        
        # Note: pm4py might default to left-to-right. We can try to adjust parameters if needed.
        # But 'rankdir': 'TB' is standard graphviz.
        parameters = {bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg"}
        gviz = bpmn_visualizer.apply(bpmn_graph, parameters=parameters)
        
        # Hack to force Top-to-Bottom if pm4py doesn't expose it easily in high level API
        # but typically Graphviz defaults are okay or we can modify the body.
        gviz.attr(rankdir='TB') 
        
        return gviz, activities, bpmn_graph

    except Exception as e:
        print(f"Error generating BPMN: {e}")
        # Return a simple error graph or None
        import graphviz
        err_g = graphviz.Digraph()
        err_g.node('Error', f"Error generating BPMN: {str(e)}")
        # Return None for the BPMN object so we don't crash downstream
        return err_g, [], None

def generate_colored_bpmn(bpmn_graph, alignments, accepted_deviations=None):
    """
    Generates a color-coded BPMN visualization based on alignment diagnostics.
    Accepted deviations are rendered as red nodes connected to the nearest formal activity.

    Args:
        bpmn_graph (pm4py.objects.bpmn.obj.BPMN): The reference BPMN model.
        alignments (list): List of alignment dictionaries from pm4py.
        accepted_deviations (set|None): Labels that the user accepted as formal variants.

    Returns:
        tuple: (graphviz.Digraph, dict) - The colored graph and compliance info map.
    """
    from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
    import graphviz

    if not bpmn_graph or not alignments:
        return None, {}

    if accepted_deviations is None:
        accepted_deviations = set()

    # 1. Analyze Alignments to count moves
    model_move_counts = {} # Activity -> Count of "Model Move Only" (Skipped)
    log_move_counts = {}   # Activity -> Count of "Log Move Only" (Deviation)
    sync_move_counts = {}  # Activity -> Count of "Sync" (Match)

    for align in alignments:
        trace_alignment = align['alignment']
        for log_move, model_move in trace_alignment:
            is_log_skip = (log_move is None or log_move == '>>')
            is_model_skip = (model_move is None or model_move == '>>')

            if not is_log_skip and not is_model_skip:
                if log_move == model_move:
                    sync_move_counts[model_move] = sync_move_counts.get(model_move, 0) + 1
            elif is_log_skip and not is_model_skip:
                model_move_counts[model_move] = model_move_counts.get(model_move, 0) + 1
            elif not is_log_skip and is_model_skip:
                log_move_counts[log_move] = log_move_counts.get(log_move, 0) + 1

    # 1b. Find the nearest preceding formal activity for each deviation
    deviation_anchors = {}  # deviation_label -> nearest_formal_label
    for align in alignments:
        last_sync = None
        for log_move, model_move in align['alignment']:
            is_log_skip = (log_move is None or log_move == '>>')
            is_model_skip = (model_move is None or model_move == '>>')
            if not is_log_skip and not is_model_skip and log_move == model_move:
                last_sync = model_move
            elif not is_log_skip and is_model_skip:
                if log_move in accepted_deviations and last_sync and log_move not in deviation_anchors:
                    deviation_anchors[log_move] = last_sync

    # 2. Generate Base Graphviz Object
    parameters = {bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg"}
    gviz = bpmn_visualizer.apply(bpmn_graph, parameters=parameters)

    # 3. Build compliance info map for UI display
    compliance_info = {}

    # 4. Apply Colors, build compliance info, and extract label->node_id mapping
    new_body = []
    label_to_node = {}
    for line in gviz.body:
        if 'label=' in line and 'shape=' in line:
            try:
                # Extract node_id (first quoted string on the line)
                nid_start = line.find('"') + 1
                nid_end = line.find('"', nid_start)
                node_id = line[nid_start:nid_end] if nid_start > 0 and nid_end > nid_start else None

                start_quote = line.find('label="') + 7
                end_quote = line.find('"', start_quote)
                label = line[start_quote:end_quote]

                if node_id and label:
                    label_to_node[label] = node_id

                fill_color = "white"
                pen_width = "1"
                color = "black"

                syncs = sync_move_counts.get(label, 0)
                skips = model_move_counts.get(label, 0)

                total = syncs + skips
                if total > 0:
                    if skips > syncs:
                        fill_color = "#f0f0f0"
                        color = "grey"
                        pen_width = "2"
                        style = "dashed,filled"
                        compliance_info[label] = "skipped"
                    else:
                        fill_color = "#e6ffe6"
                        color = "green"
                        pen_width = "2"
                        style = "filled"
                        compliance_info[label] = "executed"
                else:
                    style = "filled"

                line = line.rstrip(';\n')
                line = line.rstrip(']')
                line += f', style="{style}", fillcolor="{fill_color}", color="{color}", penwidth="{pen_width}"];'

            except:
                pass

        new_body.append(line)

    gviz.body = new_body
    gviz.attr(rankdir='TB')

    # Add log-only moves (deviations/shadow workflows) to compliance info
    for label, count in log_move_counts.items():
        if label not in compliance_info and label is not None and label != ">>":
            if label in accepted_deviations:
                compliance_info[label] = "accepted"
            else:
                compliance_info[label] = "deviation"

    # 5. Add accepted deviations as red nodes connected to their nearest formal activity
    for i, dev_label in enumerate(sorted(accepted_deviations)):
        dev_node_id = f"accepted_shadow_{i}"
        # Escape quotes in label for graphviz
        safe_label = dev_label.replace('"', '\\"')
        # Red-filled node with bold border
        gviz.body.append(
            f'\t"{dev_node_id}" [label="{safe_label}" shape=box '
            f'style="filled,bold" fillcolor="#fde8e8" color="#dc2626" '
            f'penwidth="2.5" fontcolor="#991b1b"];'
        )
        # Connect with dashed red edge from the nearest formal activity
        anchor_label = deviation_anchors.get(dev_label)
        if anchor_label and anchor_label in label_to_node:
            anchor_node_id = label_to_node[anchor_label]
            gviz.body.append(
                f'\t"{anchor_node_id}" -> "{dev_node_id}" '
                f'[style=dashed color="#dc2626" penwidth="1.5" '
                f'arrowhead=open constraint=false];'
            )

    return gviz, compliance_info

def convert_to_event_log(df):
    """
    Converts a pandas DataFrame with 'timestamp' and 'activity_name' to a PM4Py Event Log.
    Assumes all events belong to a single case 'Meeting_1'.
    """
    if df.empty:
        return None

    # Create a copy to avoid modifying original
    log_df = df.copy()
    
    # Assign a single case ID
    log_df['case:concept:name'] = 'Meeting_1'
    
    # Map columns to PM4Py standard names
    log_df['concept:name'] = log_df['activity_name']
    
    # Convert timestamp string (MM:SS or HH:MM:SS) to datetime objects for PM4Py
    # We use a dummy date for reference
    def parse_time(t_str):
        try:
            parts = str(t_str).split(":")
            if len(parts) == 2:
                m, s = int(parts[0]), int(parts[1])
                total_seconds = m * 60 + s
            elif len(parts) == 3:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                total_seconds = h * 3600 + m * 60 + s
            else:
                return datetime.now()
            return datetime(2023, 1, 1) + timedelta(seconds=total_seconds)
        except:
            return datetime.now()
            
    log_df['time:timestamp'] = log_df['timestamp'].apply(parse_time)
    
    return log_df

def generate_discovered_bpmn(log_df):
    """
    Generates a BPMN graph from an event log using Inductive Miner.
    Returns both the graphviz object AND an evidence map for UI rendering.
    """
    if log_df is None or log_df.empty:
        return None, {}
        
    try:
        # Create a mapping of Activity -> Evidence Details from the log
        activity_evidence_map = {}
        if 'details' in log_df.columns and 'activity_name' in log_df.columns:
            for _, row in log_df.iterrows():
                source = row.get('source', 'Unknown')
                detail = row.get('details', 'No details')
                clean_detail = str(detail).replace('"', "'")
                activity_evidence_map[row['activity_name']] = f"Source ({source}): {clean_detail}"

        # Discover BPMN model
        process_tree = pm4py.discover_process_tree_inductive(log_df)
        bpmn_model = pm4py.convert_to_bpmn(process_tree)
        
        # Visualize
        gviz = pm4py.visualization.bpmn.visualizer.apply(bpmn_model)
        
        return gviz, activity_evidence_map
    except Exception as e:
        st.error(f"Error generating BPMN: {e}")
        return None, {}
