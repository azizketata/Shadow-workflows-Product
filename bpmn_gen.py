import openai
import pm4py
from pm4py.objects.bpmn.obj import BPMN
import json
import tempfile
import os
import pandas as pd
import streamlit as st
from datetime import datetime

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
        
        prompt = f"""
        Extract a sequential list of activities/topics from the following meeting agenda.
        Return ONLY a JSON array of strings, where each string is a concise activity label.
        Ensure the order matches the agenda.
        
        Agenda:
        {agenda_text}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts process activities from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        # robust parsing logic in case of extra text
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("Could not find JSON array in response")
            
        activities = json.loads(content[start_idx:end_idx])
        
        # Build BPMN using PM4Py 2.7.x API
        # In recent versions, we add nodes directly to the graph using add_node() and add_flow()
        bpmn_graph = BPMN()
        
        # Create nodes
        start_event = BPMN.StartEvent(name="Start")
        bpmn_graph.add_node(start_event)
        
        end_event = BPMN.EndEvent(name="End")
        bpmn_graph.add_node(end_event)
        
        previous_node = start_event
        
        for i, act_label in enumerate(activities):
            task = BPMN.Task(name=act_label)
            bpmn_graph.add_node(task)
            
            # Add flow from previous node to current task
            flow = BPMN.SequenceFlow(previous_node, task)
            bpmn_graph.add_flow(flow)
            
            previous_node = task
        
        # Connect last task to end event
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

def generate_colored_bpmn(bpmn_graph, alignments):
    """
    Generates a color-coded BPMN visualization based on alignment diagnostics.
    
    Args:
        bpmn_graph (pm4py.objects.bpmn.obj.BPMN): The reference BPMN model.
        alignments (list): List of alignment dictionaries from pm4py.
        
    Returns:
        tuple: (graphviz.Digraph, dict) - The colored graph and compliance info map.
    """
    from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
    import graphviz
    
    if not bpmn_graph or not alignments:
        return None, {}

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
                
    # 2. Generate Base Graphviz Object
    parameters = {bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg"}
    gviz = bpmn_visualizer.apply(bpmn_graph, parameters=parameters)
    
    # 3. Build compliance info map for UI display
    compliance_info = {}
    
    # 4. Apply Colors and build compliance info
    new_body = []
    for line in gviz.body:
        if 'label=' in line and 'shape=' in line:
            try:
                start_quote = line.find('label="') + 7
                end_quote = line.find('"', start_quote)
                label = line[start_quote:end_quote]
                
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
                return datetime.strptime(f"2023-01-01 00:{t_str}", "%Y-%m-%d %H:%M:%S")
            if len(parts) == 3:
                return datetime.strptime(f"2023-01-01 {t_str}", "%Y-%m-%d %H:%M:%S")
            return datetime.now()
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
